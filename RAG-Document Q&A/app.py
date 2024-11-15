import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

from dotenv import load_dotenv
load_dotenv()
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT']= "Q&A_Project with RAG App"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(groq_api_key=groq_api_key)

promt = ChatPromptTemplate.from_template(
    """ 
    Answer the questions strickly based on the contex.
    Please provide the most accurate responce based on the question
    <context>
    {context}
    <context>
    Question: {input}
    """
)
def create_vector_embeddings():
    if "db" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)
        st.session_state.loader = PyPDFDirectoryLoader("papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.db = FAISS.from_documents(st.session_state.split_docs,embedding=st.session_state.embeddings)


user_promt = st.text_input("Enter your query")
if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector DB is created")


import time
if user_promt:
    document_chain = create_stuff_documents_chain(model,promt)
    retriver = st.session_state.db.as_retriever()
    retiver_chain = create_retrieval_chain(retriver,document_chain)
    start = time.process_time()
    response = retiver_chain.invoke({"input":user_promt})
    print(f"Responce time:{time.process_time()- start}")
    st.write(response["answer"])


    ##Streamlit expander
    with st.expander("Document similiarity Search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------")