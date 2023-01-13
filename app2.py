import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import recommender_project
from recommender_project import filtered_df, recommend_nmf
from st_aggrid import AgGrid
import pickle
import random


## add a text

df= filtered_df.T
my_set = set(df.index)
type(my_set)
l = []
for i in my_set:
    l.append(i)

st.title( "🧄 Rotten Garlics movie recommender")
st.write("please rate the movies below and click 'get recommendations'")

user_query = {}
movies_seen = st.multiselect("Please select and rate movies you have already seen", l)



for movie in movies_seen:
    ratings = st.slider(label=f'{movie}', min_value=0.0, max_value=5.0, step=0.5, key=movie)
    user_query[movie] = ratings

    
    
if st.button("Get recommendations"):
    st.header("Your best bets")
    with open('nmf_model1.pkl','rb') as file:
        loaded_model = pickle.load(file)
    st.markdown(recommend_nmf(user_query))
#  df=recommender_project.recommend_nmf(user_query,model=loaded_model)
# AgGrid(df)
 
#     #st.image("penguins.png")

    



   


# #dict={"Avatar (2009)": avatar,
# #"Zodiac (2007)":zodiac,
# #"2001: A Space Odyssey (1968)":space_Odyssey,
# #"40-Year-Old Virgin, The (2005)":virgin,
# #"X-Men: First Class (2011)":xmen
# #}


# with open('nmf_model1.pkl','rb') as file:
#     loaded_model = pickle.load(file)

# second_button= st.button("Get recommendations")
# if second_button:
#     st.header("Your best bets")
#   #  st.write( str(recommender_project.recommend_nmf(dict,model=recommender_project.loaded_model))
#     df=recommender_project.recommend_nmf(dict,model=loaded_model)[0].reset_index()
#     AgGrid(df)

#     #st.image("penguins.png")
