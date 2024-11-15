import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

#langchain tracing
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = "true"

#langchain prompt
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful AI assistance.Please respond to user queries"),
        ("user","Question: {input}")
    ]
)

##
def generate_responce(question,api_key,llm,temperature,max_token):
    openai.api_key=api_key
    model = ChatOpenAI(model=llm,temperature=temperature,max_tokens=max_token)
    output_parser = StrOutputParser()
    chain = prompt|model|output_parser
    answer = chain.invoke({"input":question})
    return answer

##Title of the Streamlit app
st.title("Q&A chat bot with Open AI")

## Sidebar for controling settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Open AI api key",type="password")

## Select the OpenAi model
llm = st.sidebar.selectbox("Select OpenAi model",["gpt-4o","gpt-4-turbo","gpt-4"])

## Adjust responce parameter
temprature =st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_token = st.sidebar.slider("Max Token",min_value = 50, max_value =500, value=300)

## Define main interface
st.write("Ask any question")
input = st.text_input("You:")

if input:
    responce = generate_responce(api_key=api_key,question=input,max_token=max_token,temperature=temprature,llm=llm)
    st.write(responce)
else:
    st.write("Please provide an input")