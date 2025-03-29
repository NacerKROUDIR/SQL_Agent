import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.database.db_manager import DatabaseManager
from src.agent.sql_agent import SQLAgent
from src.utils.visualization import generate_visualization
from src.utils.ollama_utils import get_ollama_models, get_model_display_name, get_model_by_display_name

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SQL Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "db_manager" not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if "sql_agent" not in st.session_state:
    st.session_state.sql_agent = SQLAgent()
if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = []
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "show_tables_after_init" not in st.session_state:
    st.session_state.show_tables_after_init = False
if "sql_executor_result" not in st.session_state:
    st.session_state.sql_executor_result = None

def update_sidebar_tables():
    """Update the sidebar tables session state"""
    if st.session_state.sql_agent.agent_executor:
        st.session_state.tables_for_sidebar = st.session_state.sql_agent.get_tables()
    else:
        st.session_state.tables_for_sidebar = None

def get_full_schema():
    """Get full database schema with tables and columns"""
    if not st.session_state.sql_agent.agent_executor:
        return None
    
    schema = {}
    if st.session_state.tables_for_sidebar and st.session_state.tables_for_sidebar.get("result_df") is not None:
        tables = st.session_state.tables_for_sidebar["result_df"]["table_name"]
        for table in tables:
            columns_df = get_table_columns(table)
            if columns_df is not None:
                schema[table] = [
                    {"name": row['column_name'], "type": row['data_type']} 
                    for _, row in columns_df.iterrows()
                ]
    return schema

def process_user_query(query: str) -> None:
    """Process user query using the SQL agent and display results."""
    try:
        # Get the full schema to provide as context
        schema = get_full_schema()
        
        # Get the agent response
        agent_response = st.session_state.sql_agent.process_query(query, schema)
        
        return agent_response
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def show_tables():
    """Show all tables in the database"""
    if not st.session_state.sql_agent.agent_executor:
        st.error("Please initialize the SQL Agent first")
        return False
        
    tables_result = st.session_state.sql_agent.get_tables()
    if tables_result:
        st.write(tables_result["answer"])
        with st.expander("View SQL Query"):
            st.code(tables_result["sql"], language="sql")
        if tables_result["result_df"] is not None:
            st.write("Tables:")
            st.dataframe(tables_result["result_df"])
        return True
    else:
        st.error("Failed to retrieve tables")
        return False

def sql_executor_tab():
    """Content for the SQL Executor tab"""
    st.header("SQL Executor")
    
    # Direct SQL execution
    direct_sql = st.text_area("Enter SQL Query:", height=100, key="sql_input")
    execute_button = st.button("Execute SQL")
    
    # First check if we already have results in session state
    if st.session_state.sql_executor_result:
        result = st.session_state.sql_executor_result
        st.write(result["answer"])
        
        if result["result_df"] is not None:
            st.write("Results:")
            st.dataframe(result["result_df"])
            
            # Generate visualization if possible
            generate_visualization(result["result_df"])
            
            # Add CSV download button with on_click handler
            csv = result["result_df"].to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="query_results.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        # Add clear results button
        if st.button("Clear Results"):
            st.session_state.sql_executor_result = None
            st.session_state.sql_input = ""
            st.rerun()
    
    # Only execute query if no results exist or execute button is pressed
    elif execute_button:
        if not st.session_state.sql_agent.agent_executor:
            st.error("Please initialize the SQL Agent first")
        elif not direct_sql.strip():
            st.error("Please enter a SQL query")
        else:
            result = st.session_state.sql_agent.direct_execute_query(direct_sql)
            if result:
                # Store in session state
                st.session_state.sql_executor_result = result
                
                # Display results
                st.write(result["answer"])
                
                if result["result_df"] is not None:
                    st.write("Results:")
                    st.dataframe(result["result_df"])
                    
                    # Generate visualization if possible
                    generate_visualization(result["result_df"])
                    
                    # Add CSV download button
                    csv = result["result_df"].to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
            else:
                st.error("Failed to execute SQL query")

def ai_agent_tab():
    """Content for the AI Agent tab"""
    st.header("AI Agent")
    
    # Chat history container (push this to the top)
    chat_container = st.container()
    
    # Create a container at the bottom for the input
    input_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input at the bottom
    with input_container:
        if prompt := st.chat_input("Ask me to generate or execute SQL queries..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)

                # Add assistant response
                with st.chat_message("assistant"):
                    if not st.session_state.sql_agent.agent_executor:
                        st.error("Please initialize the SQL Agent in the sidebar first")
                    else:
                        # Create a placeholder for the response
                        response_placeholder = st.empty()
                        
                        with st.spinner("Generating response..."):
                            # Get the full schema to provide as context
                            schema = get_full_schema()
                            
                            # Get the agent response
                            agent_response = st.session_state.sql_agent.process_query(prompt, schema)
                        
                        if agent_response:
                            # 1. Display the natural language response first
                            response_placeholder.write(agent_response["answer"])
                            
                            # 2. Show results table if available
                            if agent_response.get("result_df") is not None:
                                st.write("**Results:**")
                                st.dataframe(agent_response["result_df"])
                                
                                # 3. Generate visualization if possible
                                generate_visualization(agent_response["result_df"])
                            
                            # 4. Show SQL query and other details in an expander
                            with st.expander("View Details", expanded=False):
                                if agent_response.get("sql"):
                                    st.write("**SQL Query:**")
                                    st.code(agent_response["sql"], language="sql")
                            
                            # 5. Add CSV download button (only if we have results)
                            if agent_response.get("result_df") is not None:
                                csv = agent_response["result_df"].to_csv(index=False)
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name="query_results.csv",
                                    mime="text/csv"
                                )
                        else:
                            response_placeholder.error("Failed to generate a response. Please try again.")

def get_table_columns(table_name):
    """Get columns for a specific table"""
    if not st.session_state.sql_agent.agent_executor:
        return None
        
    columns_query = f"""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = '{table_name}'
    ORDER BY ordinal_position
    """
    
    # Execute query in silent mode
    result = st.session_state.sql_agent.direct_execute_query(columns_query)
    return result["result_df"] if result and result.get("result_df") is not None else None

def main():
    st.title("🤖 SQL Agent")
    st.write("An intelligent SQL query generator and executor powered by LLMs")

    # Initialize tables state if not exists
    if "tables_for_sidebar" not in st.session_state:
        st.session_state.tables_for_sidebar = None
    if "expanded_tables" not in st.session_state:
        st.session_state.expanded_tables = set()

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Database Configuration
        st.subheader("Database Settings")
        db_type = st.selectbox("Database Type", ["postgresql", "mysql"])
        db_host = st.text_input("Host", value="localhost")
        db_port = st.text_input("Port", value="5432" if db_type == "postgresql" else "3306")
        db_name = st.text_input("Database Name")
        db_user = st.text_input("Username")
        db_password = st.text_input("Password", type="password")
        
        if st.button("Initialize Database"):
            if st.session_state.db_manager.initialize(db_type, db_host, db_port, db_name, db_user, db_password):
                st.session_state.db_manager = st.session_state.db_manager
        
        # LLM Configuration
        st.subheader("LLM Settings")
        if not st.session_state.ollama_models:
            st.session_state.ollama_models = get_ollama_models()
        
        if st.session_state.ollama_models:
            # Create display names for the dropdown
            model_display_names = [get_model_display_name(model) for model in st.session_state.ollama_models]
            selected_display_name = st.selectbox("Select Model", model_display_names)
            
            # Get the selected model details
            selected_model = get_model_by_display_name(selected_display_name, st.session_state.ollama_models)
            
            # Show API key input for OpenAI models
            if selected_model and selected_model["type"] == "openai":
                st.session_state.openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    value=st.session_state.openai_api_key
                )
            
            if st.button("Initialize LLM"):
                if st.session_state.sql_agent.initialize_llm(
                    selected_model["name"],
                    selected_model["type"],
                    st.session_state.openai_api_key if selected_model["type"] == "openai" else None
                ):
                    st.session_state.sql_agent = st.session_state.sql_agent
        else:
            st.error("No models found. Please ensure Ollama is running.")
        
        # Agent Initialization
        st.subheader("Agent Settings")
        if st.button("Initialize SQL Agent"):
            if st.session_state.sql_agent.initialize_agent(st.session_state.db_manager):
                st.session_state.sql_agent = st.session_state.sql_agent
                # Update the sidebar tables immediately
                update_sidebar_tables()
        
        # Display database tables in sidebar with expandable columns
        st.subheader("Database Tables")
        if st.session_state.sql_agent.agent_executor:
            if st.button("Refresh Tables"):
                update_sidebar_tables()
                
            if st.session_state.tables_for_sidebar is None:
                update_sidebar_tables()
                
            if st.session_state.tables_for_sidebar and st.session_state.tables_for_sidebar.get("result_df") is not None:
                # Display tables with native expanders
                tables = st.session_state.tables_for_sidebar["result_df"]["table_name"]
                for table in tables:
                    with st.expander(f"📊 **{table}**"):
                        columns_df = get_table_columns(table)
                        if columns_df is not None:
                            # Create a dataframe with styled columns and data types
                            columns_html = ""
                            for _, row in columns_df.iterrows():
                                column_name = row['column_name']
                                data_type = row['data_type']
                                columns_html += f"<tr><td><b>{column_name}</b></td><td><code>{data_type}</code></td></tr>"
                            
                            html = f"""
                            <table style="width:100%">
                            <tr>
                                <th style="text-align:left">Column</th>
                                <th style="text-align:left">Type</th>
                            </tr>
                            {columns_html}
                            </table>
                            """
                            st.markdown(html, unsafe_allow_html=True)
                        else:
                            st.write("Failed to retrieve columns")
            else:
                st.write("No tables found or database not connected")
        else:
            st.write("Initialize the SQL Agent to view tables")

    # Main tabbed interface
    tab1, tab2 = st.tabs(["AI Agent", "SQL Executor"])
    
    with tab1:
        ai_agent_tab()
    
    with tab2:
        sql_executor_tab()

if __name__ == "__main__":
    main() 