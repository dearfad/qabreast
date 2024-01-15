import streamlit as st
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

llm = Tongyi()
loader = DirectoryLoader('./references/', glob="**/*.txt")
docs = loader.load()
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5')
st.write('Hello Breast!')