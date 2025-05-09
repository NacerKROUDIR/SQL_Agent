import streamlit as st
from src.utils.schema_utils import get_full_schema
from src.utils.visualization import generate_visualization
import random
import time

def ai_agent_tab():
    """Content for the AI Agent tab"""
    st.header("AI Agent")
    
    # Chat history container (push this to the top)
    chat_container = st.container()
    
    # Create a container at the bottom for the input
    input_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input at the bottom
    with input_container:
        if prompt := st.chat_input("Ask me to generate or execute SQL queries..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.write(prompt)

                # Add assistant response
                with st.chat_message("assistant"):
                    if not st.session_state.sql_agent.agent_executor:
                        st.error("Please initialize the SQL Agent in the sidebar first")
                    else:
                        # Create a placeholder for the response
                        response_placeholder = st.empty()
                        
                        with st.spinner("Generating response..."):
                            # Get the full schema to provide as context
                            schema = get_full_schema()
                            
                            # Get the agent response
                            agent_response = st.session_state.sql_agent.process_query(prompt, schema)
                        
                        if agent_response:
                            # 1. Display the natural language response first
                            response_placeholder.write(agent_response["answer"])
                            
                            # 2. Show results table if available
                            if agent_response.get("result_df") is not None:
                                st.write("**Results:**")
                                st.dataframe(agent_response["result_df"])
                                
                                # 3. Generate visualization if possible
                                generate_visualization(agent_response["result_df"])
                                
                                # 4. Add CSV download button outside expanders (with primary style and icon)
                                if not agent_response["result_df"].empty:
                                    csv = agent_response["result_df"].to_csv(index=False)
                                    # Generate a unique key for the download button with random component
                                    random_suffix = f"{random.randint(1000, 9999)}_{int(time.time() * 1000) % 10000}"
                                    button_key = f"download_csv_{len(agent_response['result_df'])}_{hash(str(agent_response['sql']))}_{random_suffix}"
                                    st.download_button(
                                        label="📥 Download Results as CSV",
                                        data=csv,
                                        file_name="query_results.csv",
                                        mime="text/csv",
                                        use_container_width=False,
                                        type="primary",
                                        key=button_key
                                    )
                            
                            # 5. Create collapsible sections for details
                            
                            # Technical Details section with SQL query
                            with st.expander("Technical Details", expanded=False):
                                if agent_response.get("sql"):
                                    st.write("**SQL Query:**")
                                    st.code(agent_response["sql"], language="sql")
                            
                            # Debug Information section - separate from technical details
                            if "debug_info" in agent_response and agent_response["debug_info"]:
                                with st.expander("Debug Information", expanded=False):
                                    for info in agent_response["debug_info"]:
                                        st.write(f"- {info}")
                                    
                                    if "intermediate_steps" in agent_response and agent_response["intermediate_steps"]:
                                        st.write("**Agent Steps:**")
                                        for i, step in enumerate(agent_response["intermediate_steps"]):
                                            if isinstance(step, tuple) and len(step) >= 2:
                                                st.write(f"**Step {i+1}:**")
                                                if hasattr(step[0], "tool"):
                                                    st.write(f"Tool: {step[0].tool}")
                                                if hasattr(step[0], "tool_input"):
                                                    st.write(f"Input: {step[0].tool_input}")
                                                st.write(f"Output: {step[1]}")
                                                st.write("---")
                        else:
                            response_placeholder.error("Failed to generate a response. Please try again.") 