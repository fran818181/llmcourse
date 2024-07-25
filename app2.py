import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch.nn.functional as F

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to load images from URLs
def load_image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)

# Sample image URLs
urls = [
    "https://images.unsplash.com/photo-1576201836106-db1758fd1c97?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=80",
    "https://images.unsplash.com/photo-1591294100785-81d39c061468?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=300&q=80",
    "https://images.unsplash.com/photo-1548199973-03cce0bbc87b?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=400&q=80"
]

# Load images from URLs
images = [load_image_from_url(url) for url in urls]
print(images)

# Function to search for the best matching image
def search_images(query, images):
    inputs = processor(text=[query], images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = F.softmax(logits_per_image, dim=0)  # Apply softmax across the images dimension
    best_image_idx = probs.argmax(dim=0).item()
    return images[best_image_idx]

# Streamlit UI
st.title("CLIP Image Search Engine")
query = st.text_input("Enter your search query:")

# Display the best matching image based on the query
if query:
    st.write("Searching for:", query)
    best_image = search_images(query, images)
    st.write("BEST MATCH:", query)
    st.image(best_image, caption="Best match", use_column_width=True)

# Display images
st.write("Available Images:")
for img in images:
    st.image(img, use_column_width=True)

