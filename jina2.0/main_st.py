import streamlit as st
from streamlit_jina import jina

st.write('Resume Semantic Search')

jina.text_search(endpoint="http://localhost:12345/search")
