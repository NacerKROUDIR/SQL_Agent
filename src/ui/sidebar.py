import streamlit as st
import pandas as pd
from src.utils.ollama_utils import get_model_display_name, get_model_by_display_name
from src.utils.schema_utils import get_table_columns, update_sidebar_tables

def render_sidebar():
    """Render the sidebar with configuration options and database tables view"""
    with st.sidebar:
        st.header("Configuration")
        
        # Database Configuration
        render_db_config()
        
        # LLM Configuration
        render_llm_config()
        
        # Agent Initialization
        render_agent_config()
        
        # Display database tables in sidebar with expandable columns
        render_tables_view()

def render_db_config():
    """Render database configuration section"""
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

def render_llm_config():
    """Render LLM configuration section"""
    st.subheader("LLM Settings")
    if not st.session_state.ollama_models:
        # Use the get_ollama_models function from session state
        st.session_state.ollama_models = st.session_state.get_ollama_models()
    
    # Check if we have any Ollama models specifically
    has_ollama_models = any(model["type"] == "ollama" for model in st.session_state.ollama_models)
    if not has_ollama_models:
        st.warning("Ollama models not found. Please ensure Ollama is running.")
    
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

def render_agent_config():
    """Render SQL agent configuration section"""
    st.subheader("Agent Settings")
    if st.button("Initialize SQL Agent"):
        if st.session_state.sql_agent.initialize_agent(st.session_state.db_manager):
            st.session_state.sql_agent = st.session_state.sql_agent
            # Update the sidebar tables immediately
            update_sidebar_tables()

def render_tables_view():
    """Render database tables and columns view"""
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