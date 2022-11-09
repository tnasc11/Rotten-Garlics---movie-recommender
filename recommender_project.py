#!/usr/bin/env python
# coding: utf-8

# ### 1.Reading in, data inputation and filter

# In[2]:


import numpy as np
from sklearn.decomposition import NMF 
import pandas as pd


# In[3]:


user_rating = pd.read_csv("/Users/tiagonascimento/spiced_projects/Repository/garlic-boosting-student-code/tiago/10_week/data/ratings.csv")
user_rating


# In[4]:


movies = pd.read_csv("/Users/tiagonascimento/spiced_projects/Repository/garlic-boosting-student-code/tiago/10_week/data/movies.csv",index_col='movieId')
movies


# In[5]:


df = pd.merge(user_rating, 
                     movies, 
                     on ='movieId', 
                     how ='left')
df


# In[6]:


df.drop(['timestamp', 'movieId','genres'], axis=1, inplace=True)


# In[7]:


df


# In[8]:


df1 = pd.pivot_table(data=df, index='userId', columns='title',values='rating')


# In[9]:


df1


# In[10]:


df1.notna().sum(axis=1)


# In[27]:


(df1.notna().sum(axis=1)>30).sum()


# In[28]:


(df1.notna().sum(axis=0)>30).sum()


# In[29]:


users_with_few_ratings = df1.notna().sum(axis=1)>30


# In[30]:


movies_with_few_ratings = df1.notna().sum(axis=0)>30


# In[38]:


filtered_df = df1.loc[users_with_few_ratings,movies_with_few_ratings]


# In[39]:


filtered_df


# In[51]:


filtered_df.mean()


# In[39]:


mean = filtered_df.mean(axis=1)


# In[40]:


mean


# In[61]:


filtered_df.mean(axis=1)


# In[40]:


filtered_df.fillna(filtered_df.mean(),axis=0,inplace=True)


# In[41]:


filtered_df


# In[42]:


filtered_df.info()


# In[43]:


filtered_df.isna().sum(axis=1)


# In[44]:


filtered_df = filtered_df.fillna(0)


# In[45]:


filtered_df.shape


# In[56]:


movies_ = filtered_df.columns.to_list()
movies_


# In[57]:


users = filtered_df.index.to_list()
users


# ### 2. Creating a model based on NMF

# #### Step 1. Create a movie-features matrix $Q$

# In[46]:


nmf_model = NMF(n_components=4, max_iter=300)


# In[47]:


nmf_model.fit(filtered_df)


# In[48]:


nmf_model.components_.shape


# In[49]:


Q_matrix = nmf_model.components_


# In[50]:


nmf_model.feature_names_in_


# In[51]:


Q = pd.DataFrame(data=Q_matrix,
            columns=nmf_model.feature_names_in_)


# In[52]:


Q


# #### Step 2. Create user-features matrix $P$

# In[54]:


nmf_model.transform(filtered_df).shape


# In[55]:


P_matrix = nmf_model.transform(filtered_df)
P_matrix


# In[58]:


pd.DataFrame(data=P_matrix,
            index = users)


# #### Step 3. Reconstruct the ratings matrix 
# $\hat{R} := P\cdot Q \sim R$

# In[59]:


P_matrix.shape,Q_matrix.shape


# In[60]:


R_hat_matrix = np.dot(P_matrix,Q_matrix)


# In[61]:


R_hat_matrix.shape


# In[62]:


R_hat =pd.DataFrame(data=R_hat_matrix,
             columns=nmf_model.feature_names_in_,
             index = users)


# In[63]:


R_hat


# In[64]:


nmf_model.reconstruction_err_


# In[67]:


error_squared = (filtered_df-R_hat)**2
error_squared.sum().sum()


# In[68]:


np.sqrt(error_squared.sum().sum())


# ### Save a Model with pickle
# The pickle module dumps an object into a binary strings

# In[70]:


import pickle

with open('nmf_model1.pkl',mode='wb') as file:
    pickle.dump(nmf_model,file)


# In[71]:


with open('nmf_model1.pkl','rb') as file:
    loaded_model = pickle.load(file)


# In[72]:


loaded_model


# #### Step 1. Receive a user query

# In[102]:


new_user_query = {"X-Men: First Class (2011)": 5,
                 "40-Year-Old Virgin, The (2005)":5,
                 "2001: A Space Odyssey (1968)":1,
                 "Zodiac (2007)":5}


# #### Step 2. Create user-feature matrix $P$ for new user

# In[103]:


new_user_dataframe =  pd.DataFrame(data=new_user_query,
            columns=movies_,
            index = ['new_user'])
new_user_dataframe


# In[104]:


new_user_dataframe_imputed = new_user_dataframe.fillna(0)
new_user_dataframe_imputed


# In[105]:


P_new_user_matrix = nmf_model.transform(new_user_dataframe_imputed)


# In[106]:


P_new_user = pd.DataFrame(data=P_new_user_matrix,
                         index = ['new_user'])


# In[107]:


P_new_user


# #### Step 3. Reconstruct the user-movie(item) matrix/dataframe for the new user
# $\hat{R}_{new-user} = P_{new-user} \cdot Q \sim R_{new-user}$

# In[109]:


R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q_matrix)


# In[110]:


R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=nmf_model.feature_names_in_,
                         index = ['new_user'])
R_hat_new_user


# #### Step 4. Get a list of k-top rated movie to recommend to the new user
# So which movies?

# In[111]:


R_hat_new_user.transpose().sort_values(by=['new_user'], ascending=False)


# In[112]:


list(new_user_query.keys())


# In[113]:


R_hat_new_user.transpose().loc[list(new_user_query.keys()),:] = 0


# In[114]:


R_hat_new_user.transpose().sort_values(by=['new_user'],ascending=False).head(4)


# ### 3. Project Task: NMF recommender function
# 1. Implement a recommender **function** that recommends movies to a new user based on the NMF model!

# In[122]:


def recommend_nmf(new_user_query, model=loaded_model, k=10):
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """
    
    recommendations = []
    
    # construct new_user-item dataframe given the query
    new_user_dataframe =  pd.DataFrame(data=new_user_query,
            columns=movies_,
            index = ['new_user'])
    new_user_dataframe_imputed = new_user_dataframe.fillna(0)

    # 2. scoring
    with open('nmf_model1.pkl','rb') as file:
        loaded_model = pickle.load(file)
    
    Q_matrix = loaded_model.components_
    
    P_new_user_matrix = loaded_model.transform(new_user_dataframe_imputed)
    
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q_matrix)
    #
    R_hat_new_user = pd.DataFrame(data=R_hat_new_user_matrix,
                         columns=nmf_model.feature_names_in_,
                         index = ['new_user'])
    
    # calculate the score with the NMF model
    
    P_new_user = pd.DataFrame(data=P_new_user_matrix,
                         index = ['new_user'])
    
    
    # 3. ranking
    
    recommendations.append(R_hat_new_user.transpose().sort_values(by=['new_user'],ascending=False).head(4))
    
    # filter out movies already seen by the user
    
    
    return recommendations


# In[123]:


recommend_nmf(new_user_query,model=loaded_model)

