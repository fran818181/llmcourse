import json
import pandas as pd
import time
from string import punctuation
from collections import Counter
from heapq import nlargest
#import nltk
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import torch
import requests
import json
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from openai.embeddings_utils import get_embedding, cosine_similarity

# Import the raw input data
# Store it in a dataframe
import pandas as pd
df_combined = pd.read_pickle('df_miami_combined.pkl')
#print(df_combined["embedding"].head())
# Ensure all embeddings are numpy arrays
#df_combined["embedding"] = df_combined["embedding"].apply(np.array,dtype=float32)

# Query the data
# Create a query function that takes the query text as input, embeds it, and
# searches the table and returns the X best hotels and the Y most relevant
# reviews of those selected hotels.

embedder = SentenceTransformer('all-mpnet-base-v2')


# Bing Image Search API credentials
bing_api_key = '1a8c256266a8410a8b6acec8905a4f11'
bing_search_url = "https://api.bing.microsoft.com/v7.0/images/search"
#bing_search_url =  "https://api.bing.microsoft.com/"

# Function to get the image URL from Bing Image Search API
def get_image_url(hotel_name):
    headers = {"Ocp-Apim-Subscription-Key": bing_api_key}
    params = {"q": hotel_name, "count": 1}  # Get only one image
    response = requests.get(bing_search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    image_url = search_results['value'][0]['contentUrl'] if 'value' in search_results and len(search_results['value']) > 0 else None
    return image_url


def search(query2, num_top_hotels, num_top_reviews_per_hotel):
  # Embed the query
  query_embedding = embedder.encode(query2)
  query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
    # Cosine_Similarity applied to each cell of the embedding column of the
  # df_combined dataframe and store in a new column called cosine_similarity
  # df_combined["cosine_similarity"] = df_combined["embedding"].apply(lambda x: util.cos_sim(query_embedding, x))
  df_results = df_combined.copy()
  df_results["cosine_similarity"] = df_combined["embedding"].apply(lambda x: util.cos_sim(query_embedding, x))
  #df_combined["cosine_similarity"] = df_combined["embedding"].apply(lambda x: util.cos_sim(torch.tensor(x, dtype=torch.float32), query_embedding).item())
  
  # Create a new dataframe that contains the results above ordered by
  # cosine_similarity in descending order and containing only the name, review,
  # cosine_similarity columns
      
  results = df_results.sort_values("cosine_similarity", ascending=False)[["name", "review", "cosine_similarity"]]
  #print(results)
    
  # Display them in a very concise and ordered manner.
  resultlist = []
  hlist = []
  for r in results.index:
      
      # For each individual hotel
      if results.name[r] not in hlist and len(hlist)<num_top_hotels:
          
          # Check all msot relevant reviews (number of reviews to check is a
          # parameter of the function, they ahve been sorted earlier)
          smalldf = results.loc[results.name == results.name[r]]
          #print(smalldf.shape)
          if smalldf.shape[0] > num_top_reviews_per_hotel: # Note shape[0] is the number of rows
            smalldf = smalldf.head(num_top_reviews_per_hotel)

          image_url = get_image_url(results.name[r])
          
          resultlist.append(
          {
            "name":results.name[r],
            "score": float(smalldf.cosine_similarity.iloc[0]),
            #"rating": smalldf.rating.max(),
            "number_reviews": smalldf.shape[0],
            "image_url": image_url, 
            "relevant_reviews": [ "REVIEW " + str(s) + ": " + smalldf.review[s] for s in smalldf.index]
          })
          hlist.append(results.name[r])
  return resultlist

# Streamlit UI
st.title("Hotel Review Search Engine")
query_num_hotels = st.selectbox("How many hotel results do you want:", [1, 2, 3, 4, 5])
query_num_reviews = st.selectbox("How many reviews per hotel do you want:", [1, 2, 3, 4, 5])
query = st.text_input("Enter your search query:")

# Display the best matching results based on the query
if query:
    st.write("Searching for:", query)
    results = search(query, num_top_hotels=query_num_hotels, num_top_reviews_per_hotel=query_num_reviews)
    
    for result in results:
        cols = st.columns([1, 3])
        with cols[0]:
            if result['image_url']:
                st.image(result['image_url'], width=100)  # Display the thumbnail image
        with cols[1]:
            st.markdown(f"### {result['name']}")
            st.markdown(f"**Score:** {result['score']:.4f}")
            st.markdown(f"**Number of Reviews:** {result['number_reviews']}")
            st.markdown("**Relevant Reviews:**")
            for review in result['relevant_reviews']:
                st.markdown(f"* {review}")
        st.markdown("---")  # Divider between hotels

    # Optionally, display the results in a table format
    df_display = pd.DataFrame(results)
    st.dataframe(df_display)


