import streamlit as st
from src.utils.visualization import generate_visualization
from src.database.db_service import DatabaseService
import random
import time

def sql_executor_tab():
    """Content for the SQL Executor tab"""
    st.header("SQL Executor")
    
    # Initialize session state variables if they don't exist
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    # Handle the Clear Results action
    if "clear_results_clicked" in st.session_state and st.session_state.clear_results_clicked:
        # Reset the clear button state
        st.session_state.clear_results_clicked = False
        # Clear results
        if "sql_executor_result" in st.session_state:
            del st.session_state.sql_executor_result
        st.session_state.current_query = ""
        # We don't modify sql_input directly here
    
    # Direct SQL execution
    direct_sql = st.text_area("Enter SQL Query:", height=100, key="sql_input")
    execute_button = st.button("Execute SQL")
    
    # Execute query when button is clicked
    if execute_button:
        # Only execute if there's a query and it's different from the last executed one
        if direct_sql.strip():
            if direct_sql != st.session_state.current_query:
                st.session_state.current_query = direct_sql
                
                with st.spinner("Executing query..."):
                    result = DatabaseService.direct_execute_query(st.session_state.db_manager, direct_sql, debug_mode=False)
                
                if result:
                    # Store in session state
                    st.session_state.sql_executor_result = result
                    
                    # Display results
                    st.write(result["answer"])
                    
                    # Check if result contains an error flag
                    if "is_error" in result and result["is_error"]:
                        # Error is already displayed through the "answer" field above
                        pass
                    elif result["result_df"] is not None and not result["result_df"].empty:
                        st.write("Results:")
                        st.dataframe(result["result_df"])
                        
                        # Generate visualization if possible
                        generate_visualization(result["result_df"])
                        
                        # Add CSV download button
                        csv = result["result_df"].to_csv(index=False)
                        random_suffix = f"{random.randint(1000, 9999)}_{int(time.time() * 1000) % 10000}"
                        button_key = f"exec_download_{random_suffix}"
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name="query_results.csv",
                            mime="text/csv",
                            type="primary",
                            key=button_key
                        )
                    elif "Error" in result["answer"] or "error" in result["answer"].lower():
                        # Display error message in red
                        st.error(result["answer"])
                    elif result["result_df"] is not None and result["result_df"].empty:
                        st.info("Query executed successfully but returned no rows.")
                    
                    # Show query details in expander
                    with st.expander("Technical Details"):
                        st.code(result["sql"], language="sql")
                        if "debug_info" in result and result["debug_info"]:
                            st.write("Debug Information:")
                            for info in result["debug_info"]:
                                st.write(f"- {info}")
                else:
                    st.error("Failed to execute SQL query. Check the database connection.")
        else:
            st.error("Please enter a SQL query")
    
    # Add clear results button - with a callback function
    if "sql_executor_result" in st.session_state:
        # Define a callback for the clear button
        def clear_results():
            st.session_state.clear_results_clicked = True
            
        st.button("Clear Results", on_click=clear_results) 