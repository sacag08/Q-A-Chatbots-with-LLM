import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts.chat import MessagesPlaceholder
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()

from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_PROJECT']= "Q&A_Project with RAG App"
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## Set up streamlit
st.title("Conversational RAG with PDF uploads and chat history")
st.write("Upload the pdfs and chat with their content")

## Input the Groq API key
groq_api_key = st.text_input("Enter the Groq API Key",type="password")

## Check if Groq API key is provided
if groq_api_key:
    model = ChatGroq(groq_api_key=groq_api_key,model="Gemma2-9b-It")
    embeddings = HuggingFaceEmbeddings()
    ## Chat Interface
    session_id = st.text_input("Session_id",value="default_sesion")

    ## Manage session history
    if "store" not in st.session_state:
        st.session_state.store={}
    
    uploaded_file=st.file_uploader("Choose PDF files",type="pdf",accept_multiple_files=True)

    ## Process the ipdoaded files
    if uploaded_file:
        documents = []
        for uploaded_file in uploaded_file:
            temppdf = f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            pdf = loader.load()
            documents.extend(pdf)

        ## Split and Embed the document
        split_docs = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits = split_docs.split_documents(documents)
        db = Chroma.from_documents(splits,embedding=embeddings)
        retriver = db.as_retriever()

        ## Promt for history aware retriver
        contextulaized_qa_system_prompt = ( 
        "Given the chat history and latest user question"
        "which might might reference context in chat history"
        "formulate a standalone question that can be understood"
        "without chat history. Do NOT answer the question"
        "just reformulate if needed and otherwise return it as it is"
        )
        contextualize_qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextulaized_qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
        )
        history_aware_retriver = create_history_aware_retriever(model,retriver,contextualize_qa_prompt)
            
        ## Question answer prompt
        system_prompt = (
        "You are a helpfull assistant for question-answering tasks"
        "Use the following prieces of retrived contex to answer"
        "the question. If you dont know the answer, sat that you dont know"
        "Use three sentences maximum and keep the answer concize"
        "\n\n"
        "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
        )
        question_answer_chain = create_stuff_documents_chain(model,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriver,question_answer_chain)

        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            else:
                return st.session_state.store[session_id]

        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        user_input = st.text_input("You question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },
            )
            st.write(st.session_state.store)
            st.write("Assistant: ",response["answer"])
            st.write("Chat History",session_history.messages)
else:
    st.warning("Please enter Groq API key")