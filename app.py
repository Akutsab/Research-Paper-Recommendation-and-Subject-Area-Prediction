# pip install tensorflow==2.19.0
# pip install sentence_transformers==5.1.0
# pip install streamlit

# import libraries===================================
import streamlit as st
import torch
from sentence_transformers import util
import pickle
from tensorflow.keras.layers import TextVectorization # pyright: ignore[reportMissingImports]
import numpy as np
from tensorflow import keras

# load save recommendation models===================================
embeddings = pickle.load(open('embeddings.pkl','rb'))
sentences = pickle.load(open('sentences.pkl','rb'))
rec_model = pickle.load(open('rec_model.pkl','rb'))

# load save prediction models============================




# create app=========================================
st.title('Research Papers Recommendation and Subject Area Prediction App')
input_paper=st.text_input("Enter Paper Title")

def recommendation(input_paper):
    # Calculate cosine similarity scores between the embeddings of input_paper and all papers in the dataset.
    cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))

    # Get the indices of the top-k most similar papers based on cosine similarity.
    top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)

    # Retrieve the titles of the top similar papers.
    papers_list = []
    for i in top_similar_papers.indices:
        papers_list.append(sentences[i.item()])

    return papers_list
if st.button("Recommend"):
    # recommendation part
    recommend_papers = recommendation(input_paper)
    st.subheader("Recommended Papers")
    st.write(recommend_papers)

     #========prediction part
    


st.markdown(
    """
    <hr>
    <div style="text-align: center; color: gray; font-size: 14px;">
        Built with ❤️ by Akshay ❤️
    </div>
    """,
    unsafe_allow_html=True
)