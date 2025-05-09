import streamlit as st
from src.database.db_service import DatabaseService

def update_sidebar_tables():
    """Update the sidebar tables session state"""
    if st.session_state.sql_agent.agent_executor:
        st.session_state.tables_for_sidebar = DatabaseService.get_tables(st.session_state.db_manager, debug_mode=False)
    else:
        st.session_state.tables_for_sidebar = None

def get_full_schema():
    """Get full database schema with tables and columns"""
    if not st.session_state.sql_agent.agent_executor:
        return None
    
    return DatabaseService.get_full_schema(st.session_state.db_manager, debug_mode=False)

def get_table_columns(table_name):
    """Get columns for a specific table"""
    if not st.session_state.sql_agent.agent_executor:
        return None
    
    # Create the SQL query directly to avoid debugging output
    columns_query = f"""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = '{table_name}'
    ORDER BY ordinal_position
    """
    
    # Execute query with silent mode
    result = DatabaseService.direct_execute_query(st.session_state.db_manager, columns_query, debug_mode=False)
    return result["result_df"] if result and result.get("result_df") is not None else None

def show_tables():
    """Show all tables in the database"""
    if not st.session_state.sql_agent.agent_executor:
        st.error("Please initialize the SQL Agent first")
        return False
        
    tables_result = DatabaseService.get_tables(st.session_state.db_manager, debug_mode=False)
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