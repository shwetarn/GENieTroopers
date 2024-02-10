from langchain.callbacks import get_openai_callback
from langchain.utilities import SQLDatabase
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
import time
import pandas as pd
import sqlite3
from langchain.chains import create_sql_query_chain

openai_api_key = "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"

df1 = pd.read_csv(r"your_train_df.csv")
print(df1.head(10))

conn = sqlite3.connect(r'second_gen_ai.db')
df1.to_sql("SECOND DATA", conn, index=True, if_exists='replace')

db = SQLDatabase.from_uri(f"second_gen_ai.db")
chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model="text-embedding-ada-002"), db)

with get_openai_callback() as cb:
    query = "Most common Issue Type in the records?"
    
    chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0,model="gpt-3.5-turbo"), db)
    response = chain.invoke({"question":query})
    print("Response",response,"\n")
    llm = OpenAI(openai_api_key=openai_api_key, temperature=0, verbose=True)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    ans = db_chain.run(query)
    print("Answer-- ::",ans,"\n")
    
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")