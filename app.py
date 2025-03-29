import os
import streamlit as st
from dotenv import load_dotenv

# Import UI components
from src.ui.ai_agent_tab import ai_agent_tab
from src.ui.sql_executor_tab import sql_executor_tab
from src.ui.sidebar import render_sidebar

# Import utilities
from src.utils.session_state import initialize_session_state

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SQL Agent",
    page_icon="🤖",
    layout="wide"
)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Set up the UI
    st.title("🤖 SQL Agent")
    st.write("An intelligent SQL query generator and executor powered by LLMs")

    # Render sidebar
    render_sidebar()

    # Main tabbed interface
    tab1, tab2 = st.tabs(["AI Agent", "SQL Executor"])
    
    with tab1:
        ai_agent_tab()
    
    with tab2:
        sql_executor_tab()

if __name__ == "__main__":
    main() 