import streamlit as st
from langchain_community.llms import Tongyi
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

llm = Tongyi()
loader = DirectoryLoader('./references/', glob="**/*.txt")
docs = loader.load()
st.write(docs[0][:200])