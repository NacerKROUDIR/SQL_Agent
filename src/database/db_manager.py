from sqlalchemy import create_engine, text
from langchain_community.utilities import SQLDatabase
from typing import Optional
import pandas as pd
import streamlit as st
import re

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.db = None

    def initialize(self, db_type: str, db_host: str, db_port: str, 
                  db_name: str, db_user: str, db_password: str) -> bool:
        """Initialize database connection using provided parameters."""
        try:
            if not all([db_name, db_user, db_password]):
                st.error("Please fill in all database credentials")
                return False

            connection_string = f"{db_type}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            self.engine = create_engine(connection_string)
            # Create LangChain SQLDatabase instance
            self.db = SQLDatabase(self.engine)
            st.success("Database connection established successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to connect to database: {str(e)}")
            return False

    def clean_query(self, query: str, silent: bool = False) -> str:
        """Clean the SQL query before execution."""
        if not query:
            return ""
            
        # Debug print - removed for all operations
        
        # Remove markdown code block syntax
        query = re.sub(r'```sql\s*', '', query)
        query = re.sub(r'```\s*$', '', query)
        # Remove any remaining backticks
        query = re.sub(r'`', '', query)
        # Remove any leading/trailing whitespace and newlines
        query = query.strip()
        # Remove any leading/trailing semicolons
        query = re.sub(r'^;+|;+$', '', query)
        
        # Handle unbalanced quotes that might cause SQL errors
        single_quotes = query.count("'")
        double_quotes = query.count('"')
        
        # More robust handling of unbalanced quotes
        if single_quotes % 2 != 0:
            # Don't output warnings, just fix silently
            # Try to find unclosed single quote and balance it
            in_quote = False
            balanced_query = ""
            for char in query:
                if char == "'":
                    in_quote = not in_quote
                balanced_query += char
            
            # If we're still in a quote at the end, add a closing quote
            if in_quote:
                balanced_query += "'"
            query = balanced_query
        
        if double_quotes % 2 != 0:
            # Don't output warnings, just fix silently
            # Try to find unclosed double quote and balance it
            in_quote = False
            balanced_query = ""
            for char in query:
                if char == '"':
                    in_quote = not in_quote
                balanced_query += char
            
            # If we're still in a quote at the end, add a closing quote
            if in_quote:
                balanced_query += '"'
            query = balanced_query
        
        # Debug print - removed for all operations
        
        return query.strip()

    def execute_query(self, query: str, silent: bool = False) -> Optional[pd.DataFrame]:
        """Execute a SQL query and return results as a DataFrame."""
        try:
            if not query or not query.strip():
                if not silent:
                    st.error("Empty query received")
                return None
            
            # Clean the query before execution
            cleaned_query = self.clean_query(query, silent=silent)
            
            if not cleaned_query:
                if not silent:
                    st.error("Query is empty after cleaning")
                return None
                
            # Debug print - removed for all operations
                
            with self.engine.connect() as connection:
                try:
                    result = connection.execute(text(cleaned_query))
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    
                    # Debug output moved to debug_info in DatabaseService
                    return df
                except Exception as e:
                    # Provide more detailed error message
                    if not silent:
                        st.error(f"SQL execution error: {str(e)}")
                        if "syntax error" in str(e).lower():
                            st.error(f"Syntax error in query: {cleaned_query}")
                    
                    # Store the error in the session state for reference
                    if hasattr(st, 'session_state'):
                        st.session_state.last_sql_error = str(e)
                    
                    # Instead of just raising the exception, return a special DataFrame with error information
                    error_df = pd.DataFrame()
                    error_df.attrs['error'] = str(e)  # Store error in DataFrame attributes
                    error_df.attrs['is_error'] = True
                    return error_df
        except Exception as e:
            if not silent:
                st.error(f"Error executing query: {str(e)}")
                st.write("Query that caused error:", query)
            
            # Store the error in the session state for reference
            if hasattr(st, 'session_state'):
                st.session_state.last_sql_error = str(e)
                
            # Return a DataFrame with error information
            error_df = pd.DataFrame()
            error_df.attrs['error'] = str(e)
            error_df.attrs['is_error'] = True
            return error_df

    def get_db(self):
        """Get the LangChain SQLDatabase instance."""
        return self.db 