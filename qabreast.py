import streamlit as st
from langchain_community.llms import Tongyi
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

@st.cache_resource(show_spinner=True)
def get_chain():
    llm = Tongyi()
    loader = DirectoryLoader('./references/', glob="**/*.txt")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:<context>{context}</context> Question: {input}""")
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

st.title('乳腺疾病专业问答')
st.caption('参考：2022版中国乳腺癌随诊随访与健康管理指南')
if prompt := st.text_input('请输入你的问题：', '子宫内膜增厚的标准是什么？'):
    answer_placeholder = st.empty()
    with st.spinner(text="增强检索中..."):
        retrieval_chain = get_chain()
        response = retrieval_chain.invoke({"input": prompt})
    answer_placeholder.write(response["answer"])