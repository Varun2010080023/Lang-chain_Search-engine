import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Page configuration for better UI
st.set_page_config(
    page_title="LangChain Smart Search Assistant",
    page_icon="🔎",
    layout="wide"
)

# App title and description
st.title("🔎 LangChain - Smart Search Assistant")
st.markdown("""
This app uses LangChain's agent framework with multiple search tools (Web Search, Arxiv, Wikipedia)
to provide comprehensive answers to your questions. Watch the agent's thought process as it searches for information!
""")

# Sidebar configuration
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your Groq API Key:", type="password", value=os.getenv("GROQ_API_KEY", ""))
    
    st.subheader("Model Settings")
    model_option = st.selectbox(
        "Select Groq Model",
        ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"]
    )
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1,
                            help="Lower values make responses more deterministic, higher values more creative")
    
    max_tokens = st.slider("Max Tokens", min_value=256, max_value=4096, value=1024, step=256,
                           help="Maximum length of the response")
    
    max_iterations = st.slider("Max Search Iterations", min_value=1, max_value=10, value=5, step=1,
                              help="Maximum number of tool use iterations")
    
    st.subheader("Tools to Use")
    use_web_search = st.checkbox("Web Search (DuckDuckGo)", value=True)
    use_arxiv = st.checkbox("Academic Papers (Arxiv)", value=True)
    use_wikipedia = st.checkbox("Wikipedia", value=True)
    
    st.markdown("---")
    st.markdown("Built with LangChain + Streamlit")

# Configure search tools
tools = []

# Add selected tools based on user preferences
if use_web_search:
    search = DuckDuckGoSearchRun(name="Web_Search", description="Useful for searching the web for current information")
    tools.append(search)

if use_arxiv:
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=1000,
        load_max_docs=2,
        load_all_available_meta=True
    )
    arxiv = ArxivQueryRun(
        api_wrapper=arxiv_wrapper,
        name="Arxiv_Search",
        description="Useful for searching academic papers on Arxiv"
    )
    tools.append(arxiv)

if use_wikipedia:
    wiki_wrapper = WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=1000
    )
    wiki = WikipediaQueryRun(
        api_wrapper=wiki_wrapper,
        name="Wikipedia_Search",
        description="Useful for searching general knowledge information on Wikipedia"
    )
    tools.append(wiki)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a search assistant that can look up information from the web, Wikipedia, and academic papers. What would you like to know?"}
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input(placeholder="Ask me anything, like 'What are the latest developments in fusion energy?'"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    # Validate API key
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    elif not tools:
        st.error("Please select at least one search tool in the sidebar.")
    else:
        try:
            # Display a spinner while processing
            with st.spinner("Searching for information..."):
                # Initialize LLM
                llm = ChatGroq(
                    groq_api_key=api_key,
                    model_name=model_option,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=True
                )
                
                # Create agent with improved configuration
                search_agent = initialize_agent(
                    tools, 
                    llm, 
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                    handle_parsing_errors=True,
                    max_iterations=max_iterations,
                    early_stopping_method="generate",
                    verbose=True
                )
                
                # Process the query
                with st.chat_message("assistant"):
                    # Container for agent's thoughts
                    thought_container = st.container()
                    
                    # Use the StreamlitCallbackHandler to show the agent's thoughts
                    st_cb = StreamlitCallbackHandler(thought_container, expand_new_thoughts=True)
                    
                    # Create a clear prompt with instructions to avoid loops
                    enhanced_prompt = f"""You are a helpful search assistant with access to several information sources.
                    
Answer the following question step by step using the provided tools.
Be direct, concise, and informative in your final answer.
If you can't find a relevant answer after 2-3 search attempts, summarize what you've learned and acknowledge limitations.
Always cite your sources in the final answer.
Avoid repeating the same search with identical parameters.

User Question: {prompt}"""
                    
                    # Run the agent
                    try:
                        start_time = time.time()
                        response = search_agent.run(enhanced_prompt, callbacks=[st_cb])
                        end_time = time.time()
                        
                        # Add response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Show response and search stats
                        st.write(response)
                        st.caption(f"⏱️ Search completed in {round(end_time - start_time, 2)} seconds")
                        
                    except Exception as agent_error:
                        st.error(f"The agent encountered an issue: {str(agent_error)}")
                        # Provide a fallback response
                        fallback_response = "I encountered an issue while searching for information. This might be due to tool errors or complexity in the question. Could you try rephrasing your question or selecting different search tools?"
                        st.session_state.messages.append({"role": "assistant", "content": fallback_response})
                        st.write(fallback_response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your API key and try again.")