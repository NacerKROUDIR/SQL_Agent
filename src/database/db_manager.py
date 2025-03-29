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
            
        # Debug print (only if not silent)
        if not silent:
            st.write("DatabaseManager received query:", query)
            
        # Remove markdown code block syntax
        query = re.sub(r'```sql\s*', '', query)
        query = re.sub(r'```\s*$', '', query)
        # Remove any remaining backticks
        query = re.sub(r'`', '', query)
        # Remove any leading/trailing whitespace and newlines
        query = query.strip()
        # Remove any leading/trailing semicolons
        query = re.sub(r'^;+|;+$', '', query)
        
        # Debug print (only if not silent)
        if not silent:
            st.write("DatabaseManager cleaned query:", query)
        
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
                
            # Debug print (only if not silent)
            if not silent:
                st.write("Executing query:", cleaned_query)
                
            with self.engine.connect() as connection:
                result = connection.execute(text(cleaned_query))
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as e:
            if not silent:
                st.error(f"Error executing query: {str(e)}")
                st.write("Query that caused error:", query)
            return None

    def get_db(self):
        """Get the LangChain SQLDatabase instance."""
        return self.db 