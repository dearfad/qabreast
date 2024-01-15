import streamlit as st
from langchain_community.llms import Tongyi
from langchain.document_loaders import DirectoryLoader
llm = Tongyi()
loader = DirectoryLoader('./references/', glob="**/*.txt")
docs = loader.load()
st.write(docs[0][:200])