# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 09:03:51 2023

@author: SHRI
"""

import pandas as pd
anime=pd.read_csv('anime.csv')
anime.shape
#you will get 12294*7 matrix

anime.columns
anime.genre
#here we are considering  only genre
from sklearn.feature_extraction.text import TfidfVectorizer
#this is term frequency inverse document
#Each row is treated as documents
tfidf=TfidfVectorizer(stop_words='english')
#it is going to create a TfidfVectorizer to seperate all stop words.
#it is going to seperate out all the words from the row
#now let us try to check null value

anime['genre'].isnull().sum()
#there are 62 null value
#suppose the one movie has got genre,Drama , Romance,....
#There may be many empty spaces
#so let us impute  these empty spaces,general is like simple imputer
anime['genre']=anime['genre'].fillna('general')
#now let us create a tfidf matrix
tfidf_matrix=tfidf.fit_transform(anime.genre)
tfidf_matrix.shape
# you will get 12294,47
#its has created sparse matrix means ,
#we want to do items based recommendation, if a user has
#watched gather, then you recommend shershah movie

from sklearn.metrics.pairwise import linear_kernel
#this is for measuring similarity
cosine_sim_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf matrix is compared
#with each element of tfidf matrix only
#ouput will be similarity matrixof size 12294X12294 size
#ere in cosine sin matrix.
#there are no movie names only index are provided
#we will try to sap movie name with movie index given
#for that purpose custon function is written 
anime_index = pd.Series(anime.index, index=anime['name']).drop_duplicate()
# we are converting amine_index into series format, we want index and correspond
anime_id=anime_index['Assassis (1995)']
anime_id
def get_recommendations (Name, topN):
    #topN=10
    #Name= 'Assassins (1995)'
    anime_id=anime_index[Name]
    

    #we want to capture whole row of given words
    #name, its score and columns and columns id
    #for that purpose we are applying cosin_sim_matrix to enumerate function
    #Enumerate function crete an object
    #which we need to create a list form
    #we  are using enumerate function,
    #what enumerate does , suppose we are given
    #(2,10,15,18), if we applay enumerate then it will create a list
    #(0,2,  1,10,   3,15,   4,18)
    cosine_scores=list(enumerate(cosine_sim_matrix[anime_id]))
    #The cosine score captured , we want to arrange in Descending order
    #so that
    #we can recommend the top 10 basedd hight similarity i.e score
    #if we will check the cosine score, it comprises the index: cosine score
    #x[0]=index and x[1] is cosine score
    # we want to arrange the tupples according to descending order of the score not index
    #Sorting the cosine_similarity  scores based on scores i.ex[1]
    
    cosine_scores=sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    #Get the score of Top N most similar movies
    # To Capture top N movies , you need to give TopN+1
    cosine_scores_N=cosine_scores[0: topN+1]
    #getting the movie index
    anime_idx=[i[0] for i in cosine_scores_N]
    #getting the cosine score
    anime_scores=[i[0] for i in cosine_scores_N]
    #we are going to use this information to create a Dataframe
    #create a empty dataframe
    anime_similar_show=pd.DataFrame(columns=['name','score'])
    #assign anime_idx to name columns
       












