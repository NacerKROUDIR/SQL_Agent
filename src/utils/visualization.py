import pandas as pd
import plotly.express as px
import streamlit as st
import hashlib
import random
import time

def generate_visualization(df: pd.DataFrame) -> None:
    """Generate appropriate visualizations based on the DataFrame content."""
    if df is None or df.empty:
        return

    # Generate a unique key based on the dataframe content plus random component
    # Use the first few rows and columns to create a hash
    sample_data = str(df.head(3).values)
    # Add a random component and timestamp to ensure uniqueness even for identical DataFrames
    random_suffix = f"{random.randint(1000, 9999)}_{int(time.time() * 1000) % 10000}"
    unique_key = f"{hashlib.md5(sample_data.encode()).hexdigest()}_{random_suffix}"

    # Try to identify numeric columns for visualization
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) >= 2:
        # Create a scatter plot if we have at least 2 numeric columns
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        # Remove default annotations about the SQL query
        fig.update_layout(
            annotations=[]
        )
        # Add unique key to plotly_chart
        st.plotly_chart(fig, key=f"scatter_{unique_key}")
    elif len(numeric_cols) == 1:
        # Create a bar chart if we have one numeric column
        fig = px.bar(df, y=numeric_cols[0])
        # Remove default annotations about the SQL query
        fig.update_layout(
            annotations=[]
        )
        # Add unique key to plotly_chart
        st.plotly_chart(fig, key=f"bar_{unique_key}") 