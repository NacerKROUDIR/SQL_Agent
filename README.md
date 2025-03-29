# SQL Agent 🤖

An intelligent SQL query generator and executor powered by Ollama LLMs, built with Streamlit.

## Features

- 🤖 Natural language to SQL query generation
- 🔄 Multi-database support (PostgreSQL, MySQL)
- 📊 Interactive data visualization with Plotly
- 📥 CSV export functionality
- 🔒 Secure credential management
- ⚡ Query caching and timeout controls

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- PostgreSQL or MySQL database
- Git

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sql-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` with your database credentials and Ollama configuration.

## Running the Application

1. Ensure Ollama is running locally:
```bash
ollama serve
```

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. Click "Initialize Database" in the sidebar to establish database connection
2. Click "Initialize LLM" to set up the Ollama model
3. Start chatting with the SQL Agent in natural language
4. View generated SQL queries, execution results, and visualizations
5. Download results as CSV when available

## Configuration

The application can be configured through the `.env` file:

- Database settings (type, host, port, credentials)
- Ollama model selection and parameters
- Query timeout and cache settings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 