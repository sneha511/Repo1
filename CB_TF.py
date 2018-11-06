import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings; warnings.simplefilter('ignore')
import pandas.io.sql as psql
import mysql.connector
import numpy as np

# Use user, password, host and datbase name for mysql connection
conn = mysql.connector.connect(user='root', password='1234',
                              host='127.0.0.1',
                              database='pb')
cursor= conn.cursor()

articles_df = psql.read_sql('SELECT * FROM post', conn)
#print("First five rows of post table:-\n")
#print (articles_df.head(5))

interactions_df = pd.read_csv('All-interactions.csv')
#print("First five rows of analytics_data_reader_user_events table:-\n")
#print (interactions_df.head(5))

interactions_df['eventStrength'] = interactions_df['interactions'].apply(lambda x: x > 0, lambda x: x + 10)
#interactions_df['eventStrength'] = interactions_df['interactions']

users_interactions_count_df = interactions_df.groupby(['user_id', 'post_id']).size().groupby('user_id').size()

print('# Users with interactions: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 2].reset_index()[['user_id']]
print('# Users with at least 2 interactions: %d' % len(users_with_enough_interactions_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['user_id']]
print('# Users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 100].reset_index()[['user_id']]
print('# Users with at least 100 interactions: %d' % len(users_with_enough_interactions_df))

print('# Total interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'user_id',
               right_on = 'user_id')
print('# Of interactions from users with at least 2 interactions: %d' % len(interactions_from_selected_users_df))

#To model the user interest on a given post, we aggregate all the interactions the user has performed in an item by a weighted sum of interaction type strength and apply a log transformation 

def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['user_id', 'post_id'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()

print('# Unique user/item interactions: %d' % len(interactions_full_df))
print (interactions_full_df.head(10))

print("Applying TF-IDF.....\n")
#Ignoring stopwords (words with no semantics) from English
stopwords_list = stopwords.words('english')


#Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus, ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1,2),
                     min_df=0.009,
                     max_df=0.2,
                     max_features=5000,
                     stop_words=stopwords_list)

item_ids = articles_df['id'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['content'])
tfidf_feature_names = vectorizer.get_feature_names()

# Building item profile and user profile

def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx+1]
    return item_profile

def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles

def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['post_id'])
    
    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
    #Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm

def build_users_profiles(): 
    interactions_indexed_df = interactions_full_df[interactions_full_df['post_id'] \
                                                   .isin(articles_df['id'])].set_index('user_id')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles

user_profiles = build_users_profiles()

# Applying cosine similarity between user profile and item profile

print("Applying Cosine Similarity.....\n")
class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        
    def get_model_name(self):
        return self.MODEL_NAME
        
    def _get_similar_items_to_user_profile(self, person_id, topn=10):
        #Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['id', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'id', 
                                                          right_on = 'id')[['recStrength', 'id']]           
        return recommendations_df
    
content_based_recommender_model = ContentBasedRecommender(articles_df)

print("#####################################################")
print("Content-Based Recommender System is ready....\n")
a=input("Enter a user id:")
print("\nContent-Based Recommendations for given user are:\n")

print(content_based_recommender_model.recommend_items(a, topn=5, verbose=True))

print("\n#####################################################")

cursor.close()
conn.close()
