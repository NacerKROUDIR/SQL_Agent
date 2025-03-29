import requests
from typing import List, Dict, Tuple
import streamlit as st
import os

def get_ollama_models() -> List[Dict[str, str]]:
    """Fetch available models from Ollama and add OpenAI models."""
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        response = requests.get(f"{base_url}/api/tags")
        if response.status_code == 200:
            ollama_models = [{"name": model["name"], "type": "ollama"} for model in response.json()["models"]]
            # Add OpenAI models
            openai_models = [
                {"name": "gpt-4o-mini", "type": "openai"}
            ]
            return ollama_models + openai_models
    except Exception as e:
        st.error(f"Failed to fetch Ollama models: {str(e)}")
    return []

def get_model_display_name(model: Dict[str, str]) -> str:
    """Get display name for a model."""
    return f"{model['name']} ({model['type'].upper()})"

def get_model_by_display_name(display_name: str, models: List[Dict[str, str]]) -> Dict[str, str]:
    """Get model details by display name."""
    for model in models:
        if get_model_display_name(model) == display_name:
            return model
    return None 