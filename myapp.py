import streamlit as st
import pandas as pd
import numpy as np
from streamlit_elements import elements, mui, html, dashboard
from langchain.callbacks import get_openai_callback
from langchain.utilities import SQLDatabase
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
import time
import sqlite3
from langchain.chains import create_sql_query_chain
 
openai_api_key = "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"
 
def get_response(query):
    db = SQLDatabase.from_uri(f"sqlite:///gen_ai.db")
    chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model="text-embedding-ada-002"), db)
   
    with get_openai_callback() as cb:
        chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0,model="gpt-3.5-turbo"), db)
        response = chain.invoke({"question":query})
        #print("Response",response,"\n")
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0, verbose=True)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        ans = db_chain.run(query)
        return ans
 
def chatbot():
    st.set_page_config(page_title="GENie Troopers", layout="wide")
    st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)
    st.header('Chat with your Data*', divider='rainbow')
    user_input = st.text_input("Enter your message")
   
    if "fruits" not in st.session_state:
        st.session_state.fruits = []
   
    st.session_state.get(st.session_state.fruits)
    def appendQuery():
        st.session_state.fruits.append(user_input)
       
    def appendResponse():
        answer = get_response(user_input)
        st.session_state.fruits.append(answer)
 
    if st.button('Send Message'):
        appendQuery()
        appendResponse()
    for index,x in enumerate(st.session_state.fruits):
        if index%2==0:
            user_message = str(x)
            user_str = f"""<div style='margin-top:2rem;text-align: right;'><span style='padding:0.5rem;background-color:grey;border-radius:5px;color:white;'><i class='fa-solid fa-user'></i>&nbsp;&nbsp;User Message: {user_message}</span></div>"""
            st.markdown(user_str, unsafe_allow_html=True)
        else:
            chatbot_message = str(x)
            chatbot_str = f"""<div style='margin-top:2rem;text-align: left;'><span style='padding:0.5rem;background-color:grey;border-radius:5px;color:white;'><i class='fa-solid fa-robot'></i>&nbsp;&nbsp;Message from Bot: {chatbot_message}</span></div>"""
            st.markdown(chatbot_str, unsafe_allow_html=True)
   
if __name__ == "__main__":
    chatbot()