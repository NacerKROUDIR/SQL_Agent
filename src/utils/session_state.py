import streamlit as st
from src.database.db_manager import DatabaseManager
from src.agent.sql_agent import SQLAgent
from src.utils.ollama_utils import get_ollama_models

def initialize_session_state():
    """Initialize all session state variables"""
    # Chat messages history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Database and agent instances
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    if "sql_agent" not in st.session_state:
        st.session_state.sql_agent = SQLAgent()
    
    # LLM-related state
    if "ollama_models" not in st.session_state:
        st.session_state.ollama_models = []
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""
    
    # Function for fetching models (stored in session state for mocking in tests)
    if "get_ollama_models" not in st.session_state:
        st.session_state.get_ollama_models = get_ollama_models
    
    # UI state
    if "sql_executor_result" not in st.session_state:
        st.session_state.sql_executor_result = None
    
    # Database tables state
    if "tables_for_sidebar" not in st.session_state:
        st.session_state.tables_for_sidebar = None
    if "expanded_tables" not in st.session_state:
        st.session_state.expanded_tables = set() 