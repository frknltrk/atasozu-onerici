import torch
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("dbmdz/bert-base-turkish-uncased")
embeddings = torch.load("proverb_embeddings.pt")
df = pd.read_csv("atasozleri-vk.csv", index_col=[0])

st.set_page_config(
    page_title="Türkçe Atasözü Önerici",
    page_icon="🎈",
)

st.title("Türkçe Atasözü Önerici")
st.subheader("BERT Tabanlı Atasözü Öneri Sistemi")

girdi = st.text_area(
    "Aklından geçenleri buraya dök."
)

if girdi:
    girdi_vek = model.encode(girdi)
    sim_vec = util.cos_sim(girdi_vek, embeddings)[0]
    best_indices = sim_vec.topk(5).indices
    
    results = df[['title', 'anlam']].iloc[best_indices]

    st.subheader("Öneriler")
    st.write(results)


