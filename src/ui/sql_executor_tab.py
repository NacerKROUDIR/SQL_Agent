import streamlit as st
from src.utils.visualization import generate_visualization

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