import streamlit as st
import google.generativeai as genai
import langchain
import sqlite3
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# Load the Google API Key from environment variables
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initial prompt template
prompt = [
    """
    Imagine you're an SQL expert and data visualization advisor adept at translating English questions into precise SQL queries and recommending visualization types for a database with the following schema:

    {schema}

    Here are examples to guide your query generation and visualization recommendation:

    - Example Question 1: "How many unique company names are there?"
      SQL Query: SELECT COUNT(DISTINCT name) FROM Companies;
      Recommended Chart: None (The result is a single numeric value.)

    - Example Question 2: "What are the total number of companies in each city?"
      SQL Query: SELECT city, COUNT(*) AS total_companies FROM Companies GROUP BY city;
      Recommended Chart: Bar chart (Cities on the X-axis and total companies on the Y-axis.)

    - Example Question 3: "List all companies with more than 500 employees."
      SQL Query: SELECT name FROM Companies WHERE company_size > 500;
      Recommended Chart: None (The result is a list of company names.)

    - Example Question 4: "What percentage does each formatted_work_type represent of the total?"
      SQL Query: SELECT formatted_work_type, (COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Jobs)) AS percentage FROM Jobs GROUP BY formatted_work_type;
      Recommended Chart: Pie chart (Show each formatted_work_type's percentage of the total.)

    - Example Question 5: "Which companies have the most job openings?"
      SQL Query: SELECT Companies.name, COUNT(Jobs.job_id) AS total_openings FROM Companies JOIN Jobs ON Companies.company_id = Jobs.company_id GROUP BY Companies.name ORDER BY total_openings DESC LIMIT 10;
      Recommended Chart: Bar chart (Company names on the X-axis and total job openings on the Y-axis.)

    Your task is to craft the correct SQL query in response to the given English questions and suggest an appropriate chart type for visualizing the query results, if applicable. Please ensure that the SQL code generated does not include triple backticks (```) at the beginning or end and avoids including the word "sql" within the output. Also, provide clear and concise chart recommendations when the query results lend themselves to visualization.
    """
]

# Function to extract database schema (tables and columns)
def get_database_schema(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        schema[table_name] = column_names
    
    conn.close()
    return schema

# Function to generate a prompt with the database schema
def generate_prompt_with_schema(schema):
    schema_str = "\n".join([f"Table: {table}\nColumns: {', '.join(columns)}" for table, columns in schema.items()])
    
    prompt_with_schema = prompt[0].format(schema=schema_str)
    return prompt_with_schema

# Function to get a response from the Gemini model
def get_gemini_response(question, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, question])
    return response.text

# Function to execute SQL query and fetch results into a DataFrame
def read_sql_query(sql, db):
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

# Function to extract SQL query from the model response
def get_sql_query_from_response(response):
    try:
        query_start = response.index('SELECT')  
        query_end = response.index(';') + 1  
        sql_query = response[query_start:query_end]  
        return sql_query
    except ValueError:
        st.error("Could not extract SQL query from the response.")
        return None

# Function to determine the appropriate chart type based on DataFrame columns
def determine_chart_type(df):
    if len(df.columns) == 2:
        if df.dtypes[1] in ['int64', 'float64'] and len(df) > 1:
            return 'bar'
        elif df.dtypes[1] in ['int64', 'float64'] and len(df) <= 10:
            return 'pie'
    elif len(df.columns) >= 3 and df.dtypes[1] in ['int64', 'float64']:
        return 'line'
    return None

# Function to generate the chart based on the chart type
def generate_chart(df, chart_type):
    if chart_type == 'bar':
        fig = px.bar(df, x=df.columns[0], y=df.columns[1], title=f"{df.columns[0]} vs. {df.columns[1]}",
                     template="plotly_white", color=df.columns[0],
                     labels={df.columns[0]: "Category", df.columns[1]: "Count"})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", 
                          yaxis=dict(title='Count'), 
                          xaxis=dict(title='Category'))
        fig.update_traces(marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == 'pie':
        fig = px.pie(df, names=df.columns[0], values=df.columns[1],
                     title=f"Distribution of {df.columns[0]}",
                     template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    elif chart_type == 'line':
        fig = px.line(df, x=df.columns[0], y=df.columns[1],
                      title=f"{df.columns[1]} Over {df.columns[0]}",
                      template="plotly_white", markers=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No suitable chart type determined for this data.")

# Streamlit setup
st.set_page_config(page_title="Text to SQL with Visualization", layout="wide")

st.markdown("""
    <h1 style="color: purple; text-align: center;">
        📊 Text to SQL with Visualization 📊
    </h1>
    """, unsafe_allow_html=True)

# File upload section
uploaded_file = st.file_uploader("Upload your SQLite (.sqlite) database file", type="sqlite")

if uploaded_file is not None:
    # Save uploaded file to the project directory
    with open("uploaded_database.sqlite", "wb") as f:
        f.write(uploaded_file.getbuffer())
    database_path = "uploaded_database.sqlite"
    
    st.success("Database uploaded and saved successfully.")
    
    # Display tables and data preview
    conn = sqlite3.connect(database_path)
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    st.subheader("Available Tables:")
    st.write(tables)

    st.subheader("Data Preview:")
    for table_name in tables['name']:
        st.write(f"**Table: {table_name}**")
        preview_df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 5;", conn)
        st.dataframe(preview_df)
    conn.close()

    # Question input and SQL query generation
    with st.container():
        st.subheader("What are you looking for?")
        col1, col2 = st.columns([3, 1], gap="small")

        with col1:
            question = st.text_input("Input your question here:", key="input", placeholder="Type here...")

        with col2:
            st.write("")  
            submit = st.button("Retrieve Data", help="Click to submit your question.")

    if submit and question:
        # Extract the schema and generate the updated prompt
        schema = get_database_schema(database_path)
        prompt_with_schema = generate_prompt_with_schema(schema)
        
        # Get the response from the model
        response = get_gemini_response(question, prompt_with_schema)
        sql_query = get_sql_query_from_response(response)
        
        if sql_query:
            st.code(sql_query, language='sql')
            df = read_sql_query(sql_query, database_path)
            
            if not df.empty:
                col_data, col_chart = st.columns(2)
                with col_data:
                    st.subheader("Query Results:")
                    st.dataframe(df)

                    # Add download button for exporting as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download data as CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv",
                    )
                
                chart_type = determine_chart_type(df)
                
                if chart_type:
                    with col_chart:
                        st.subheader("Visualization:")
                        generate_chart(df, chart_type)  
            else:
                st.write("No results found for the given query.")
        else:
            st.write("No valid SQL query could be extracted from the response.")
