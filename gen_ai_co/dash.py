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
            ax.text(0.5, 0.95, f"Number of tickets: {count}", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            st.pyplot(fig)
    
        
    
    def chart2():
        file_path = r"your_train_df.csv"
        df = pd.read_csv(file_path)
        df['Created'] = pd.to_datetime(df['Created'], format='%d-%m-%Y %H:%M')
    
        default_start_date = df['Created'].min()
        default_end_date = df['Created'].max()
    
        st.markdown("<h3 style='text-align: center; color: rgb(227, 38, 54);'>Count Plot on Ticket Classification</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col2:
            start_date = st.date_input("From", default_start_date, key = "chart_2a")
        with col3:
            end_date = st.date_input("To", default_end_date, key = "chart_2b")
    
        # Get unique values for Category and Sub Category
        categories = df['Custom field (Incident Category)'].unique()
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col2:
            selected_category = st.selectbox("Select Category", categories)
            sub_categories = df['Custom field (Incident Sub Category)'].unique()
            sub_categories = df[df['Custom field (Incident Category)'] == selected_category]['Custom field (Incident Sub Category)'].unique()
    
        with col3:
            selected_sub_category = st.selectbox("Select Sub Category", sub_categories)
    
        filtered_df = filter_by_date2(df, start_date, end_date, selected_category, selected_sub_category)
        plot_count_plot2(filtered_df)
    
    
    def filter_by_date1(df, from_date, to_date):
        from_date = pd.to_datetime(from_date)
        to_date = pd.to_datetime(to_date)
        return df[(df['Created'] >= from_date) & (df['Created'] <= to_date)]
    
    
    def plot_count_plot1(df):
        col1, col2 = st.columns([2,1])
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df, x='Status', ax=ax)
            st.markdown("<h3 style='text-align: center; color: rgb(227, 38, 54);'>Count Plot of Ticket Status</h3>", unsafe_allow_html=True)
            ax.set_xlabel('Status')
            ax.set_ylabel('Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            count = len(df)
            ax.text(0.5, 0.95, f"Number of data: {count}", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            st.pyplot(fig)
    
        # Display count in DataFrame format
        with col2:
            count_df = df['Status'].value_counts().reset_index()
            count_df.columns = ['Status', 'Count']
            st.write("Count Table:")
            st.dataframe(count_df)
    
    def chart1():
        file_path = r"your_train_df.csv"
        df = pd.read_csv(file_path)
        df['Created'] = pd.to_datetime(df['Created'], format='%d-%m-%Y %H:%M')
    
        default_start_date = df['Created'].min()
        default_end_date = df['Created'].max()
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            start_date = st.date_input("From", default_start_date, key = "chart_1a")
        with col3:
            end_date = st.date_input("To", default_end_date, key = "chart_1b")
    
        filtered_df1 = filter_by_date1(df, start_date, end_date)
        plot_count_plot1(filtered_df1)
    
    def chart():
        st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>', unsafe_allow_html=True)
        st.header('Dashboard *(for defect analysis)*', divider='rainbow')
        # st.header('Trend Analytics of Tickets Classification', divider='rainbow')
        chart1()
        chart2()
    chart()
