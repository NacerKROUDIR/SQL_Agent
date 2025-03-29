import streamlit as st

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