from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
import streamlit as st
from typing import Optional, Dict, Any, List, Union
import re

class MyChatOpenAI(ChatOpenAI):
    def _generate(self, *args, **kwargs):
        res = super()._generate(*args, **kwargs)
        res.generations[0].text = res.generations[0].text.strip("'")
        return res
    
class SQLAgent:
    def __init__(self):
        self.llm = None
        self.agent_executor = None
        self.db_manager = None
        self.sql_chain = None

    def initialize_llm(self, model_name: str, model_type: str, api_key: Optional[str] = None) -> bool:
        """Initialize LLM with selected model."""
        try:
            if model_type == "ollama":
                self.llm = Ollama(
                    base_url="http://localhost:11434",
                    model=model_name,
                    temperature=0.7,
                    num_ctx=2048,
                    num_thread=4,
                    repeat_penalty=1.1
                )
            elif model_type == "openai":
                if not api_key:
                    st.error("OpenAI API key is required for OpenAI models")
                    return False
                
                # Test the API key validity before initializing
                try:
                    import openai
                    client = openai.OpenAI(api_key=api_key)
                    
                    # Make a simple API call to validate the key
                    response = client.models.list()
                    
                    # If we get here, the API key is valid
                    self.llm = MyChatOpenAI(
                        model_name=model_name,
                        temperature=0.7,
                        openai_api_key=api_key
                    )
                except Exception as e:
                    if "authentication" in str(e).lower() or "api key" in str(e).lower():
                        st.error("Invalid OpenAI API key")
                    else:
                        st.error("Error connecting to OpenAI")
                    return False
            else:
                st.error(f"Unsupported model type: {model_type}")
                return False

            st.success(f"LLM initialized successfully with model: {model_name} ({model_type.upper()})")
            return True
        except Exception as e:
            st.error(f"Failed to initialize LLM: {str(e)}")
            return False

    def clean_sql_query(self, query: str) -> str:
        """Clean the SQL query by removing markdown code block syntax and other problematic characters."""
        if not query:
            return ""
            
        # Remove markdown code block syntax
        query = re.sub(r'```sql\s*', '', query)
        query = re.sub(r'```\s*$', '', query)
        
        # Remove any remaining backticks
        query = re.sub(r'`', '', query)
        
        # Remove any leading/trailing whitespace and newlines
        query = query.strip()
        
        # Remove any leading/trailing semicolons
        # query = re.sub(r'^;+|;+$', '', query)
        
        # Handle escaped quotes
        # query = query.replace('\\"', '"')
        
        # Handle unbalanced quotes that might cause SQL errors
        # Count single and double quotes
        """
        single_quotes = query.count("'")
        double_quotes = query.count('"')
        
        # If there's an odd number of quotes, remove the last one
        if single_quotes % 2 != 0 and query.endswith("'"):
            query = query[:-1]
        if double_quotes % 2 != 0 and query.endswith('"'):
            query = query[:-1]
        """
            
        return query.strip()

    def execute_sql_tool(self, query: str) -> str:
        """Tool for executing SQL queries.
        
        Args:
            query (str): The SQL query to execute. Should be a raw SQL query without any surrounding quotes.
                For example: SELECT * FROM table_name
                Not: ```sql SELECT * FROM table_name``` or "SELECT * FROM table_name"
        
        Returns:
            str: The query results as a string, or an error message if execution failed.
        """
        print("#"*100)
        print(query)
        if not self.db_manager:
            return "Error: Database manager not initialized"
            
        # Handle empty input
        if not query or query.strip() == "" or query.strip() == "()":
            return "Error: Please provide a valid SQL query. Example: SELECT * FROM table_name"
            
        # Clean the query
        query = self.clean_sql_query(query)
        
        # Use DatabaseService with debug mode enabled for agent operations
        from src.database.db_service import DatabaseService
        
        result = DatabaseService.direct_execute_query(self.db_manager, query, debug_mode=True)
        
        if not result:
            return "Error executing query or no results returned"
        
        # Check if there was an SQL error
        if "is_error" in result and result["is_error"]:
            # Return the detailed error message to help the agent understand what went wrong
            error_message = result["answer"]
            
            # Store query information for tracking
            st.session_state.last_query = result["sql"]
            st.session_state.last_query_error = error_message
            
            return f"Error: {error_message}"
            
        # Store query information for tracking
        st.session_state.last_query = result["sql"]
        
        if result["result_df"] is None or result["result_df"].empty:
            return "Query executed successfully but returned no results"
            
        # Store the result in session state for later use
        st.session_state.last_query_result = result["result_df"]
        
        # Convert result to string for agent
        result_str = result["result_df"].to_string(index=False)
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "\n... [truncated, showing first 2000 characters]"
            
        return result_str

    def get_schema_tool(self, _="") -> str:
        """Tool to get the database schema."""
        if not self.db_manager:
            return "Error: Database manager not initialized"
            
        try:
            # Use DatabaseService with debug mode for agent operations
            from src.database.db_service import DatabaseService
            
            # Get the full schema
            schema = DatabaseService.get_full_schema(self.db_manager, debug_mode=True)
            
            if not schema:
                return "No tables found in database or error fetching schema"
                
            # Format the result for the agent
            result = "Database Schema:\n"
            
            for table, columns in schema.items():
                result += f"\nTable: {table}\n"
                for col in columns:
                    result += f"  - {col['name']} ({col['type']})\n"
            
            return result
        except Exception as e:
            return f"Error fetching schema: {str(e)}"

    def initialize_agent(self, db_manager) -> bool:
        """Initialize the agent with SQL tools and chains."""
        try:
            if not self.llm:
                st.error("Please initialize the LLM first")
                return False
                
            self.db_manager = db_manager
                
            # Define tools
            tools = [
                Tool(
                    name="execute_sql_query",
                    func=self.execute_sql_tool,
                    description="""Useful for when you need to execute a SQL query. 
                    Input should be a raw SQL query WITHOUT any surrounding quotes or function call syntax.
                    Example: SELECT * FROM table_name
                    NOT: ```sql SELECT * FROM table_name``` or "SELECT * FROM table_name"
                    """
                ),
                Tool(
                    name="get_schema",
                    func=self.get_schema_tool,
                    description="Useful for when you need to understand the database schema. Returns tables and their columns. No input is required for this tool."
                )
            ]
            
            # Define prompt template for the agent
            prompt_template = """You are a SQL expert that helps users query databases. 
            
            When asked a question, think through what tables and columns you need to query, then use the provided tools to execute SQL queries and return the results.
            
            Always explain your reasoning in a clear, step-by-step manner. First understand what the user is asking for, then decide what SQL query would get that information, then execute it.
            
            {schema_context}
            
            IMPORTANT SQL QUERY GUIDELINES:
            1. ALWAYS quote column and table names with double quotes (e.g., "teamID" not teamID)
            2. For string matching, put string literals in single quotes: WHERE name = 'example'
            3. For aggregate functions like COUNT, AVG, SUM, etc., make sure to use an alias for readability
            
            For your final answer, provide:
            1. A natural language explanation of the results
            2. DO NOT include the SQL query in your answer - it will be displayed separately in the UI
            
            Tools:
            {tools}
            
            Use the following format:
            
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action (must be a string)
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: your final answer with explanation ONLY - do not include the SQL query
            
            IMPORTANT: When using tools, you must provide ONLY the tool name without any parentheses or function call syntax. For example, use "get_schema" not "get_schema()".
            
            Begin!
            
            Question: {input}
            {agent_scratchpad}"""
            
            prompt = PromptTemplate.from_template(prompt_template)
            
            # Create agent
            agent = create_react_agent(self.llm, tools, prompt)
            
            # Create agent executor with simpler, more compatible configuration
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                return_intermediate_steps=True,
                max_iterations=10,
            )
            
            st.success("SQL Agent initialized successfully!")
            return True
        except Exception as e:
            st.error(f"Failed to initialize SQL Agent: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

    def extract_sql_from_text(self, text: str) -> Optional[str]:
        """Extract SQL query from text."""
        # Look for SQL query in markdown format
        sql_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL)
        if sql_match:
            return self.clean_sql_query(sql_match.group(1))
        
        # Look for SQL query without markdown but with SELECT keyword
        sql_match = re.search(r'(?i)(SELECT\s+.*?)(;|\Z)', text, re.DOTALL)
        if sql_match:
            return self.clean_sql_query(sql_match.group(1))

        # Look for a pattern where SQL is preceded by like "SQL query:" or "SQL:"
        sql_match = re.search(r'(?i)(?:SQL(?:\s+query)?:?)\s*(.*?)\s*(?:$|\n\n)', text, re.DOTALL)
        if sql_match:
            return self.clean_sql_query(sql_match.group(1))
            
        return None

    def process_query(self, query: str, schema: Optional[Dict[str, List[Dict[str, str]]]] = None) -> Optional[Dict[str, Any]]:
        """Process a user query using the agent"""
        try:
            if not self.agent_executor:
                st.error("Please initialize the SQL Agent first")
                return None
            
            # Clear previous results
            if 'last_query' in st.session_state:
                del st.session_state.last_query
            if 'last_query_result' in st.session_state:
                del st.session_state.last_query_result
            
            # Add debugging info
            st.session_state.debug_info = []
            
            # Prepare input with schema context if available
            input_dict = {"input": query}
            if schema:
                schema_str = "Database Schema:\n"
                for table, columns in schema.items():
                    schema_str += f"\nTable: {table}\n"
                    for col in columns:
                        schema_str += f"  - {col['name']} ({col['type']})\n"
                
                input_dict["schema_context"] = schema_str
                st.session_state.debug_info.append("Using provided schema context")
            else:
                input_dict["schema_context"] = ""

            # Then, use the agent for additional context and execution
            try:
                agent_response = self.agent_executor(input_dict)
            except Exception as agent_error:
                st.error(f"Agent execution error: {str(agent_error)}")
                if hasattr(agent_error, "__traceback__"):
                    import traceback
                    st.error(traceback.format_exception(None, agent_error, agent_error.__traceback__))
                return {
                    "answer": "I encountered an error while processing your query. Please try again with a simpler query or check the agent settings.",
                    "sql": "",
                    "result_df": None,
                    "intermediate_steps": [],
                    "debug_info": st.session_state.debug_info
                }
            
            # Extract SQL query from the response or intermediate steps
            sql_query = None
            
            # First try to find SQL in intermediate steps
            if "intermediate_steps" in agent_response:
                for step in agent_response["intermediate_steps"]:
                    if isinstance(step, tuple) and len(step) >= 2:
                        action = step[0]
                        if hasattr(action, "tool") and action.tool == "execute_sql":
                            if hasattr(action, "tool_input"):
                                sql_query = self.clean_sql_query(action.tool_input)
                                break
            
            # If not found in steps, try to extract from final answer
            if not sql_query and "output" in agent_response:
                sql_query = self.extract_sql_from_text(agent_response["output"])
            
            # Use last query from session state if still not found
            if not sql_query:
                sql_query = st.session_state.get("last_query", "")
            
            # Combine responses from both chain and agent
            answer = ""
            answer += agent_response.get("output", "No answer provided")
            
            # Remove common SQL explanations from the answer
            sql_explanations = [
                r"The SQL query used to obtain this information is:[\s\S]*?(?=\n\n|\Z)",
                r"The SQL query I used is:[\s\S]*?(?=\n\n|\Z)",
                r"Here's the SQL query I used:[\s\S]*?(?=\n\n|\Z)",
                r"SQL query:[\s\S]*?(?=\n\n|\Z)",
                r"```sql[\s\S]*?```"
            ]
            
            for pattern in sql_explanations:
                answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)
            
            # Remove any double newlines resulting from the removals
            answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)
            answer = answer.strip()
            
            # Prepare the result
            result = {
                "answer": answer,
                "sql": sql_query,
                "result_df": st.session_state.get("last_query_result", None),
                "intermediate_steps": agent_response.get("intermediate_steps", []),
                "debug_info": st.session_state.debug_info
            }
            
            return result
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None 