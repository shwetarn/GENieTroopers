import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from streamlit_elements import elements, mui, html, dashboard
from langchain.callbacks import get_openai_callback
from langchain.utilities import SQLDatabase
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
import time
import sqlite3
import seaborn as sns
from langchain.chains import create_sql_query_chain
from snowflake.snowpark import Session

openai_api_key = "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"

# Function to filter data based on date range
def filter_by_date(df, from_date, to_date):
    return df[(df['Created'] >= from_date) & (df['Created'] <= to_date)]

# Function to plot count plot
def plot_count_plot(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='Category')
    plt.title('Count Plot of Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    st.pyplot()


def dashboard():
    listofColumns = ["In Progress", "Resolved", "Escalated","Waiting for support", "Closed"]
    st.header('Dashboard *(for defect analysis)*', divider='rainbow')
    st.markdown("""
            <style>
                    .block-container {
                        padding-top: 3rem;
                        padding-bottom: 0rem;
                        padding-left: 2rem;
                        padding-right: 2rem;
                    }
            </style>
            """, unsafe_allow_html=True)

    st.markdown("""---""")

    chart_data = pd.DataFrame(
        {"Incident": np.random.randn(20), "Service": np.random.randn(20)}
    )
    # plot_count_plot1(filtered_df1)

    firstRow, firstRow2 = st.columns([3, 1])
    # df = pd.DataFrame(data)
    firstRow2.dataframe(chart_data)
    firstRow.bar_chart(
        chart_data, y=["Incident", "Service"], color=["#FF0000", "#0000FF"]  # Optional
    )


    col1, Incident, Service = st.columns(3)

    with col1:
        st.header("A cat")

    with Incident:
        st.header("A dog")

    with Service:
        st.header("An owl")

    col1, Incident = st.columns([3, 1])
    data = np.random.randn(10, 1)

    col1.subheader("A wide column with a chart")
    col1.line_chart(data)

    Incident.subheader("A narrow column with the data")
    Incident.write(data)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
    sizes = [15, 30, 45, 10]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)

    data = {
        'ctry': ['USA', 'PHI', 'CHN'],
        'gold': [12, 1, 20,],
        'silver': [4,4, 12],
        'bronze': [8, 2, 30],
        'sum': [24, 7, 62]
    }

    df = pd.DataFrame(data)
    st.dataframe(df)
    medal_type = st.selectbox('Medal Type', ['gold', 'silver', 'bronze'])
    col1_1, Incident_1 = st.columns([3, 1])
    # filtered_df1 = filter_by_date1(df, start_date, end_date)
    
    with col1_1:
        st.write(df)
        st.write("df")
        st.write("medal_type")
        st.write(medal_type)
        fig = px.pie(df, values=medal_type, names='ctry',
                        title=f'percentage of {medal_type} medals',
                        height=300, width=200)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
        st.plotly_chart(fig, use_container_width=True)

    Incident_1.write("Number of "+str(medal_type)+" medals")
    Incident_1.write("China: " + str(data[medal_type][2]))
    Incident_1.write("USA: "+str(data[medal_type][0]))
    Incident_1.write("Philip: "+str(data[medal_type][1]))
    # st.table(df)

    count_df = df['Status'].value_counts().reset_index()
    count_df.columns = ['Status', 'Count']
    st.write("Count Table:")
    st.dataframe(count_df)
    # df = pd.DataFrame(data)
    # st.dataframe(df)

        
def get_response(query):
    db = SQLDatabase.from_uri(f"sqlite://///workspaces/codespaces-blank/gen_ai_co/second_gen_ai.db")
    chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model="text-embedding-ada-002"), db)
   
    with get_openai_callback() as cb:
        chain = create_sql_query_chain(ChatOpenAI(openai_api_key=openai_api_key, temperature=0,model="gpt-3.5-turbo"), db)
        response = chain.invoke({"question":query})
        #print("Response",response,"\n")
        llm = OpenAI(openai_api_key=openai_api_key, temperature=0, verbose=True)
        db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
        ans = db_chain.run(query)
        return ans
 
def snowflakeI():
    @st.cache_resource
    def create_session():
        return Session.builder.configs(st.secrets.snowflake).create()
    session = create_session()
    st.success("Connected to Snowflake..❄️")

    @st.cache_data
    def load_data(table_name):
        ## Read in data table
        sql = f"select * from snowflake_sample_data.tpch_sf1.lineitem limit 20"
        st.write(f"Here's some example data from `{table_name}`:")
        table = session.table(table_name)
        
        ## Do some computation on it
        table = table.limit(100)
        
        ## Collect the results. This will run the query and download the data
        table = table.collect()
        return table
    
    # Select and display data table
    table_name = "GENIE.PUBLIC.genforhack"
    
    ## Display data table
    with st.expander("See Table"):
        df = load_data(table_name)
        st.dataframe(df)
    
    ## Writing out data
    # for row in df:
    #     st.write(f"{row[0]} has a :{row[1]}:")

def chatbot():
    @st.cache_resource
    def create_session():
        return Session.builder.configs(st.secrets.snowflake).create()
    session = create_session()
    st.success("Connected to Snowflake..❄️")
    

    st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)
    st.header('GenBOT, Defect Classifier ☃️', divider='rainbow')
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

def dash():

    def filter_by_date2(df, from_date, to_date, category, sub_category):
    
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
        filtered_df = df[(df['Created'] >= from_date) & (df['Created'] <= to_date)]
        filtered_df = filtered_df[filtered_df['Custom field (Incident Category)'] == category]
        filtered_df = filtered_df[filtered_df['Custom field (Incident Sub Category)'] == sub_category]
        return filtered_df
    
    def plot_count_plot2(df):
        col1, col2, col3 = st.columns([1,1, 1])
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df, x='Custom field (Incident Category)', hue='Custom field (Incident Sub Category)', ax=ax)
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        
            # Add annotation for count of rows
            count = len(df)
            st.write(f"Number of tickets: {count}")
            nplist = [[count]]
            nplist = np.array(nplist)
            chart_data = pd.DataFrame(nplist, columns=["a"])
            st.scatter_chart(chart_data)
            # ax.text(0.5, 0.95, f"Number of tickets: {count}", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            # st.pyplot(fig)
    
        
    
    def chart2():
        file_path = r"your_train_df.csv"
        df = pd.read_csv(file_path)
        df['Created'] = pd.to_datetime(df['Created'], format='%d-%m-%Y %H:%M')
    
        default_start_date = df['Created'].min()
        default_end_date = df['Created'].max()
    
        st.markdown("<br><br><b>Count Plot on Ticket Classification</b>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        categories = df['Custom field (Incident Category)'].unique()
        with col1:
            start_date = st.date_input("From", default_start_date, key = "chart_2a")
        with col2:
            end_date = st.date_input("To", default_end_date, key = "chart_2b")
        with col3:
            selected_category = st.selectbox("Select Category", categories)
            sub_categories = df['Custom field (Incident Sub Category)'].unique()
            sub_categories = df[df['Custom field (Incident Category)'] == selected_category]['Custom field (Incident Sub Category)'].unique()
    
        with col4:
            selected_sub_category = st.selectbox("Select Sub Category", sub_categories)
    
        # Get unique values for Category and Sub Category
        # col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    
        filtered_df = filter_by_date2(df, start_date, end_date, selected_category, selected_sub_category)
        plot_count_plot2(filtered_df)
    
    
    def filter_by_date1(df, from_date, to_date):
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
        return df[(df['Created'] >= from_date) & (df['Created'] <= to_date)]
    
    
    def plot_count_plot1(df):
        col1, col2 = st.columns([2,1])
        with col1:
            # count_df = df['Status'].value_counts().reset_index()
            # nplist = []
            # for x in range(len((count_df)['count'])):
            #     temp = [count_df['count'][x]]
            #     nplist.append(temp)
            # nplist = np.array(nplist)
            # chart_data = pd.DataFrame(nplist, columns=["a"])
            # st.line_chart(chart_data)
            # count_df.columns = ['Ticket Status', 'Count']

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df, x='Status', ax=ax)
            ax.set_xlabel('Status')
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            count = len(df)
            ax.text(0.5, 0.95, f"Number of data: {count}", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            st.pyplot(fig)
            # fig = px.pie(df, values=ax, names='ctry',
            #             title=f'percentage of  medals',
            #             height=300, width=200)
            # fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
            # st.plotly_chart(fig, use_container_width=True)
    
        # Display count in DataFrame format
        with col2:
            count_df = df['Status'].value_counts().reset_index()
            # st.write("Count Table:")
            st.dataframe(count_df)
    
    def chart1():
        file_path = r"your_train_df.csv"
        df = pd.read_csv(file_path)
        df['Created'] = pd.to_datetime(df['Created'], format='%d-%m-%Y %H:%M')
    
        default_start_date = df['Created'].min()
        default_end_date = df['Created'].max()
        col1, col2 = st.columns([1, 1])
        # with col1:
        #     st.markdown("<div style='display:flex,justify-content:center;align-content:center'><b>Count Plot Ticket Status between </b></div>", unsafe_allow_html=True)
        with col1:
            start_date = st.date_input("From", default_start_date, key = "chart_1a")
        with col2:
            end_date = st.date_input("To", default_end_date, key = "chart_1b")
    
        filtered_df1 = filter_by_date1(df, start_date, end_date)
        # st.markdown("""---""")
        plot_count_plot1(filtered_df1)
    
    def chart():
        st.header('Dashboard *(for defect analysis)*', divider='rainbow')
        st.markdown("""
            <style>
                    .block-container {
                        padding-top: 3rem;
                        padding-bottom: 0rem;
                        padding-left: 2rem;
                        padding-right: 2rem;
                    }
            </style>
            """, unsafe_allow_html=True)
        @st.cache_resource
        def create_session():
            return Session.builder.configs(st.secrets.snowflake).create()
        session = create_session()
        st.success("Connected to Snowflake..❄️")
        # st.markdown("""---""")
        # st.header('Trend Analytics of Tickets Classification', divider='rainbow')
        tab1, tab2 = st.tabs(["Ticket Status", "Ticket Classification"])
        with tab1:
            chart1()
        with tab2:
            chart2()
        # col1, col2 = st.columns([2,1])
        # with col1:
        #     chart1()
        # with col2:
        #     chart2()
    chart()


def main():
    st.set_page_config(page_title="GENie Troopers", layout="wide")
    st.sidebar.markdown("<p style='text-align: left; color: #0492c2;'><u>GENie Troopers</u></p>", unsafe_allow_html=True)
    st.sidebar.title("Choose an option from below")
    page = st.sidebar.radio("Go to", ["Chatbot", "Dashboard","Snowflake Integration"])
    if page == "Dashboard":
        # dashboard()
        dash()
    if page == "Chatbot":
        chatbot()
    if page == "Snowflake Integration":
        snowflakeI()

if __name__ == "__main__":
    main()