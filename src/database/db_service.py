import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, List
from src.database.db_manager import DatabaseManager

class DatabaseService:
    """Service class to handle database operations separate from the SQL Agent."""
    
    @staticmethod
    def get_tables(db_manager: DatabaseManager, debug_mode: bool = False) -> Optional[Dict[str, Any]]:
        """Get all tables in the database."""
        try:
            if not db_manager:
                if debug_mode:
                    st.error("Database manager not initialized")
                return None
                
            # Query for PostgreSQL
            sql_query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            
            # Execute the query in silent mode
            df = db_manager.execute_query(sql_query, silent=True)
            
            if df is None:
                return {
                    "answer": "No tables found in the database.",
                    "sql": sql_query,
                    "result_df": None,
                    "intermediate_steps": [],
                    "debug_info": []
                }
                
            return {
                "answer": f"Found {len(df)} tables in the database.",
                "sql": sql_query,
                "result_df": df,
                "intermediate_steps": [],
                "debug_info": []
            }
        except Exception as e:
            if debug_mode:
                st.error(f"Error getting tables: {str(e)}")
            return None
    
    @staticmethod
    def get_table_columns(db_manager: DatabaseManager, table_name: str, debug_mode: bool = False) -> Optional[pd.DataFrame]:
        """Get columns for a specific table"""
        if not db_manager:
            return None
            
        columns_query = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' 
        AND table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        
        # Execute query with debug mode setting
        result = DatabaseService.direct_execute_query(db_manager, columns_query, debug_mode=debug_mode)
        return result["result_df"] if result and result.get("result_df") is not None else None
    
    @staticmethod
    def direct_execute_query(db_manager: DatabaseManager, sql_query: str, debug_mode: bool = False) -> Optional[Dict[str, Any]]:
        """Execute SQL query directly without using the agent."""
        try:
            # Debug: Log the original query (only in debug mode)
            if debug_mode:
                # Store in debug_info but don't display
                debug_info = [f"Original query: {sql_query}"]
            else:
                debug_info = []
            
            # Fail early if no database connection
            if not db_manager or not db_manager.engine:
                if debug_mode:
                    debug_info.append("Database not initialized or connected")
                return {
                    "answer": "Error: Database not initialized or not connected. Please initialize the database first.",
                    "sql": sql_query,
                    "result_df": None,
                    "intermediate_steps": [],
                    "debug_info": debug_info
                }
            
            # Clean the query - silent mode unless in debug
            cleaned_query = db_manager.clean_query(sql_query, silent=True)  # Always silent
            
            if debug_mode:
                debug_info.append(f"Cleaned query: {cleaned_query}")
            
            if not cleaned_query:
                if debug_mode:
                    debug_info.append("Query is empty after cleaning")
                return {
                    "answer": "Could not execute query: Empty query after cleaning",
                    "sql": sql_query,
                    "result_df": None,
                    "intermediate_steps": [],
                    "debug_info": debug_info
                }
            
            # Check if this is a SELECT/SHOW query (which should return results)
            is_select_query = cleaned_query.strip().upper().startswith(('SELECT', 'SHOW'))
                
            # Execute the query - always silent mode
            df = db_manager.execute_query(cleaned_query, silent=True)

            # Check if the DataFrame has error information
            if df is not None and hasattr(df, 'attrs') and 'is_error' in df.attrs and df.attrs['is_error']:
                error_msg = df.attrs.get('error', 'Unknown SQL error')
                if debug_mode:
                    debug_info.append(f"SQL Error: {error_msg}")
                
                # Format error message for better readability
                formatted_error = f"Error executing SQL query: {error_msg}"
                
                # Add more context for common SQL errors
                if "syntax error" in error_msg.lower():
                    formatted_error += "\nThis appears to be a syntax error. Please check your SQL syntax."
                elif "does not exist" in error_msg.lower() and "column" in error_msg.lower():
                    formatted_error += "\nThe specified column does not exist in the table."
                elif "does not exist" in error_msg.lower() and "table" in error_msg.lower():
                    formatted_error += "\nThe specified table does not exist in the database."
                
                return {
                    "answer": formatted_error,
                    "sql": cleaned_query,
                    "result_df": pd.DataFrame(),  # Return empty DataFrame
                    "intermediate_steps": [],
                    "debug_info": debug_info,
                    "is_error": True  # Add an explicit flag for error detection
                }

            if df is None:
                if debug_mode:
                    debug_info.append("Execute query returned None")
                error_msg = "Query executed but returned no results."
                
                # Enhance error message based on query type
                if is_select_query:
                    error_msg = "Query executed but returned no matching rows. The table might be empty or the WHERE condition didn't match any rows."
                
                return {
                    "answer": error_msg,
                    "sql": cleaned_query,
                    "result_df": pd.DataFrame(),  # Return empty DataFrame instead of None
                    "intermediate_steps": [],
                    "debug_info": debug_info
                }
            
            if len(df) == 0 and is_select_query:
                if debug_mode:
                    debug_info.append("Query returned zero rows")
                return {
                    "answer": "Query executed successfully, but no rows matched your criteria.",
                    "sql": cleaned_query,
                    "result_df": df,  # Return the empty DataFrame
                    "intermediate_steps": [],
                    "debug_info": debug_info
                }
            
            if debug_mode:
                debug_info.append(f"Query returned dataframe with {len(df)} rows")
                if len(df) > 0:
                    debug_info.append(f"First row sample: {df.iloc[0].to_dict()}")
                
            # Prepare the result
            return {
                "answer": f"Query executed successfully, returned {len(df)} rows.",
                "sql": cleaned_query,
                "result_df": df,
                "intermediate_steps": [],
                "debug_info": debug_info
            }
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            if debug_mode:
                import traceback
                trace_info = traceback.format_exc()
            else:
                trace_info = ""
                
            return {
                "answer": error_msg,
                "sql": sql_query,
                "result_df": pd.DataFrame(),  # Return empty DataFrame instead of None
                "intermediate_steps": [],
                "debug_info": [f"Error: {str(e)}", trace_info] if debug_mode else []
            }
    
    @staticmethod
    def get_full_schema(db_manager: DatabaseManager, debug_mode: bool = False) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """Get full database schema with tables and columns"""
        if not db_manager:
            return None
        
        schema = {}
        tables_result = DatabaseService.get_tables(db_manager, debug_mode=debug_mode)
        
        if tables_result and tables_result.get("result_df") is not None:
            tables = tables_result["result_df"]["table_name"]
            for table in tables:
                columns_df = DatabaseService.get_table_columns(db_manager, table, debug_mode=debug_mode)
                if columns_df is not None:
                    schema[table] = [
                        {"name": row['column_name'], "type": row['data_type']} 
                        for _, row in columns_df.iterrows()
                    ]
        return schema 