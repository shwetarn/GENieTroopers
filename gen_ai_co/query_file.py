from langchain.callbacks import get_openai_callback
from langchain.utilities import SQLDatabase
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
import time
import pandas as pd
import sqlite3
import streamlit as st
from langchain.chains import create_sql_query_chain

openai_api_key = "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"

if 'main_response' not in st.session_state:
    st.session_state.main_response = ""

col1, col2= st.columns([2,1])
with col1:
    query = st.text_area("Query", height=180)
with col2:
    search_button = st.button("Search")
    
if 'query' not in st.session_state:
    st.session_state.query = ""
    
if search_button:
    db = SQLDatabase.from_uri(f"second_gen_ai.db")
    chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model="text-embedding-ada-002"), db)

    with get_openai_callback() as cb:
        #query = "Most common Issue Type in the records?"
        
        chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0,model="gpt-3.5-turbo"), db)
        response = chain.invoke({"question":query})
        print("Response",response,"\n")
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0, verbose=True)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        ans = db_chain.run(query)
        print("Answer-- ::",ans,"\n")
        st.session_state.main_response = ans
        response = st.text_area("Response", value=st.session_state.main_response, height=150)

        
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")