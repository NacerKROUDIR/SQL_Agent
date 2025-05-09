# SQL Agent 🤖

An intelligent SQL query generator and executor powered by LLMs (Ollama and OpenAI), built with Streamlit. This tool allows you to interact with your database using natural language and provides a powerful SQL query execution interface.

## Features

- 🤖 Natural language to SQL query generation using LLMs (Ollama or OpenAI)
- 🔄 Support for PostgreSQL and MySQL databases
- 📊 Interactive data visualization of query results
- 📝 Direct SQL query execution with real-time feedback
- 🔍 Detailed error messages and debugging information
- 📥 Export results to CSV
- 🎯 Smart query validation and error handling
- 🔒 Secure database credential management

## Prerequisites

- Python 3.8+
- PostgreSQL or MySQL database
- [Ollama](https://ollama.ai/) installed and running locally (for Ollama models)
- OpenAI API key (optional, for OpenAI models)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SQL_Agent.git
cd SQL_Agent
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

## Running the Application

1. Start Ollama (if using Ollama models):
```bash
ollama serve
```

2. Start the Streamlit application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

## Usage

1. In the sidebar, configure your database connection:
   - Select database type (PostgreSQL/MySQL)
   - Enter connection details (host, port, database name, username, password)
   - Click "Initialize Database"

2. Configure the LLM:
   - Select a model (Ollama or OpenAI)
   - For OpenAI models, enter your API key
   - Click "Initialize LLM"

3. Initialize the SQL Agent:
   - Click "Initialize SQL Agent" in the sidebar
   - Wait for confirmation of successful initialization

4. Use the application:
   - **AI Agent Tab**: Ask questions in natural language and get SQL queries and results
   - **SQL Executor Tab**: Write and execute SQL queries directly
   - View query results, visualizations, and download data as CSV
   - Access detailed error messages and debugging information in the Technical Details expander

## Environment Variables

The application uses the following environment variables:
- `OLLAMA_BASE_URL`: URL for Ollama API (default: http://localhost:11434)

## Dependencies

Key dependencies include:
- streamlit
- langchain
- pandas
- sqlalchemy
- psycopg2-binary (for PostgreSQL)
- python-dotenv
- plotly (for visualizations)

## Error Handling

The application provides detailed error messages for:
- Database connection issues
- SQL syntax errors
- Invalid table/column references
- Query execution failures
- LLM initialization problems

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 