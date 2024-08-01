import json
import pandas as pd
import time
from string import punctuation
from collections import Counter
from heapq import nlargest
import nltk
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

          resultlist.append(
          {
            "name":results.name[r],
            "score": smalldf.cosine_similarity[r][0],
            #"rating": smalldf.rating.max(),
            "number_reviews": smalldf.shape[0],
            "relevant_reviews": [ "REVIEW " + str(s) + ": " + smalldf.review[s] for s in smalldf.index]
          })
          hlist.append(results.name[r])
  return resultlist


# Streamlit UI
st.title("CLIP Image Search Engine")
query = st.text_input("Enter your search query:")

# Display the best matching image based on the query
if query:
    st.write("Searching for:", query)
    # search(query, num_top_hotels=2, num_top_reviews_per_hotel=3) 
    st.write(search(query, num_top_hotels=2, num_top_reviews_per_hotel=3) )




