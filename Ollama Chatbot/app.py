import streamlit as st
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

#langchain tracing
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = "Ollama Chatbot"
os.environ['LANGCHAIN_TRACING_V2'] = "true"

#langchain prompt
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful AI assistance.Please respond to user queries"),
        ("user","Question: {input}")
    ]
)

##
def generate_responce(question,llm,temperature,max_token):
    model = Ollama(model=llm,temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt|model|output_parser
    answer = chain.invoke({"input":question})
    return answer

##Title of the Streamlit app
st.title("Q&A chat bot with Ollama")

## Select the OpenAi model
llm = st.sidebar.selectbox("Select Ollama model",["gemma2:2b","llama3","mistral"])

## Adjust responce parameter
temprature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_token = st.sidebar.slider("Max Token",min_value=50,max_value=500,value=150)
## Define main interface
st.write("Ask any question")
input = st.text_input("You:")

if input:
    responce = generate_responce(question=input,temperature=temprature,llm=llm,max_token=max_token)
    st.write(responce)
else:
    st.write("Please provide an input")