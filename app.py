import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from chatData import fruits


def dashboard():
   listofColumns = ["In Progress", "Resolved", "Escalated","Waiting for support", "Closed"]
   # st.title('Dashboard')
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


   # components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)
   # st.header("Dashboard *(for defect analysis)*")
   # :tulip::cherry_blossom::rose::hibiscus::sunflower::blossom:
   st.markdown("""---""")
   # "Waiting for support", "Closed"

   chart_data = pd.DataFrame(
      {"Incident": np.random.randn(20), "Service": np.random.randn(20)}
   )

   firstRow, firstRow2 = st.columns([3, 1])
   # df = pd.DataFrame(data)
   firstRow2.dataframe(chart_data)
   firstRow.bar_chart(
       chart_data, y=["Incident", "Service"], color=["#FF0000", "#0000FF"]  # Optional
   )

   # with st.sidebar:
   #     st.write("**Choose an option from below**")
   #     st.write("Chatbot")
   #     st.write("Dashboard")
   #    #  add_radio = st.radio(
   #    #      "Choose a shipping method",
   #    #      ("Standard (5-15 days)", "Express (2-5 days)")
   #    #  )

   col1, Incident, Service = st.columns(3)

   with col1:
      st.header("A cat")
      # st.image("https://static.streamlit.io/examples/cat.jpg")

   with Incident:
      st.header("A dog")
      # st.image("https://static.streamlit.io/examples/dog.jpg")

   with Service:
      st.header("An owl")
      # st.image("https://static.streamlit.io/examples/owl.jpg")


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
   with col1_1:
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

   df = pd.DataFrame(data)
   st.dataframe(df)

   cols = st.columns([3, 1])

   # with cols[0]:
   #     medal_type = st.selectbox('Medal Type', ['gold', 'silver', 'bronze'])

   #     fig = px.pie(df, values=medal_type, names='ctry',
   #                  title=f'number of {medal_type} medals',
   #                  height=300, width=200)
   #     fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
   #     st.plotly_chart(fig, use_container_width=True)

   # with cols[1]:
   #     st.text_input('sunburst', label_visibility='hidden', disabled=True)
   #     fig = px.sunburst(df, path=['ctry', 'gold', 'silver', 'bronze'],
   #                       values='sum', height=300, width=200)
   #     fig.update_layout(margin=dict(l=20, r=20, t=30, b=0),)
   #     st.plotly_chart(fig, use_container_width=True)
   from streamlit_elements import elements, mui, html
   with elements("dashboard"):
       
       from streamlit_elements import dashboard
       layout = [
           # Parameters: element_identifier, x_pos, y_pos, width, height, [item properties...]
           dashboard.Item("first_item", 0, 0, 2, 2),
           dashboard.Item("second_item", 2, 0, 2, 2, isDraggable=False, moved=False),
           dashboard.Item("third_item", 0, 2, 1, 1, isResizable=False),
       ]

       # Next, create a dashboard layout using the 'with' syntax. It takes the layout
       # as first parameter, plus additional properties you can find in the GitHub links below.

       with dashboard.Grid(layout):
           mui.Paper("First item", key="first_item")
           mui.Paper("Second item (cannot drag)", key="second_item")
           mui.Paper("Third item (cannot resize)", key="third_item")

       # If you want to retrieve updated layout values as the user move or resize dashboard items,
       # you can pass a callback to the onLayoutChange event parameter.

       def handle_layout_change(updated_layout):
           # You can save the layout in a file, or do anything you want with it.
           # You can pass it back to dashboard.Grid() if you want to restore a saved layout.
           print(updated_layout)

       with dashboard.Grid(layout, onLayoutChange=handle_layout_change):
           mui.Paper("First item", key="first_item")
           mui.Paper("Second item (cannot drag)", key="second_item")
           mui.Paper("Third item (cannot resize)", key="third_item")

def chatbot():
    st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)
    st.header('Chatbot *(for defect analysis)*', divider='rainbow')
    user_input = st.text_input("Enter your message")
    ss = st.session_state.get(fruits)
    result = 0
    def appendFuncCall():
        fruits.append(user_input)
        # fruits.append(user_input)
    if st.button('Send Message'):
        appendFuncCall()
    for index,x in enumerate(fruits):
        if index%2==0:
            bot_message = str(x)
            bot_str = f"""<div style='margin-top:2rem;text-align: left;'><span style='padding:0.5rem;background-color:grey;border-radius:5px;color:white;'><i class='fa-solid fa-robot'></i>&nbsp;&nbsp;Message from Bot: {bot_message}</span></div>"""
            st.markdown(bot_str, unsafe_allow_html=True)
            # bot_message = "Hello there! How can I help you today?"
        else:
            user_message = str(x)
            user_str = f"""<div style='margin-top:2rem;text-align: right;'><span style='padding:0.5rem;background-color:grey;border-radius:5px;color:white;'><i class='fa-solid fa-user'></i>&nbsp;&nbsp;User Message: {user_message}</span></div>"""
            st.markdown(user_str, unsafe_allow_html=True)

    # st.markdown(bot_str, unsafe_allow_html=True)
    # st.markdown(user_str, unsafe_allow_html=True)
    # st.markdown(bot_str, unsafe_allow_html=True)
    # st.markdown(user_str, unsafe_allow_html=True)
    # st.markdown(bot_str, unsafe_allow_html=True)
    # st.markdown(user_str, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="GENie Troopers", layout="wide")
    st.sidebar.title("Choose an option from below")
    page = st.sidebar.radio("Go to", ["Chatbot", "Dashboard"])
    st.markdown("<p style='text-align: left; color: #0492c2;'><u>GENie Troopers</u></p>", unsafe_allow_html=True)
    if page == "Dashboard":
        dashboard()
    if page == "Chatbot":
        chatbot()

if __name__ == "__main__":
    main()