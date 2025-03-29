import pandas as pd
import plotly.express as px
import streamlit as st

def generate_visualization(df: pd.DataFrame) -> None:
    """Generate appropriate visualizations based on the DataFrame content."""
    if df is None or df.empty:
        return

    # Try to identify numeric columns for visualization
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) >= 2:
        # Create a scatter plot if we have at least 2 numeric columns
        fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
        st.plotly_chart(fig)
    elif len(numeric_cols) == 1:
        # Create a bar chart if we have one numeric column
        fig = px.bar(df, y=numeric_cols[0])
        st.plotly_chart(fig) 