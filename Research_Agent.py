import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import requests
from bs4 import BeautifulSoup
from pydantic import Field, BaseModel

# LangChain core components
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain_ollama import OllamaLLM

# Updated LangChain core message imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages

# Initialize the Ollama LLM with Llama 3.2
def initialize_llm(model_name="llama3.2:latest", temperature=0.7):
    """Initialize the Ollama LLM with specified model."""
    return OllamaLLM(model=model_name, temperature=temperature)

# Define custom tools
class WebContentTool(BaseTool):
    name: str = "web_content_extractor"
    description: str = "Extracts and summarizes content from a provided URL"
    
    def _run(self, url: str) -> str:
        try:
            response = requests.get(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }, timeout=10)  # Added timeout for better error handling
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Truncate if too long
            if len(text) > 8000:
                text = text[:8000] + "... [content truncated]"
                
            return text
        except requests.exceptions.RequestException as e:
            return f"Error extracting content from {url}: {str(e)}"
        except Exception as e:
            return f"Unexpected error processing {url}: {str(e)}"

class QueryRouterTool(BaseTool):
    name: str = "query_router"
    description: str = "Routes the query to the appropriate tool based on the question type"
    
    def _run(self, query: str) -> str:
        llm = initialize_llm(temperature=0.2)
        
        routing_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
            Classify the following query into one of these categories:
            1. Web search - requires current information or facts
            2. Web content analysis - requires reading and analyzing a specific URL
            3. Reasoning or analysis - can be answered with existing knowledge
            
            Query: {query}
            
            Classification (just return the number):
            """
        )
        
        routing_chain = LLMChain(llm=llm, prompt=routing_prompt)
        result = routing_chain.invoke({"query": query})  # Changed from .run() to .invoke()
        
        try:
            # Extract just the number from the result
            category = int(''.join(filter(str.isdigit, result["text"].strip())))  # Access the 'text' key from the result
            if category not in [1, 2, 3]:
                category = 3  # Default to reasoning if classification is out of range
            return f"Category: {category}"
        except:
            return "Category: 3"  # Default to reasoning if classification fails

def create_research_agent():
    llm = initialize_llm()
    
    # Define tools
    search_tool = DuckDuckGoSearchRun()
    web_content_tool = WebContentTool()
    query_router_tool = QueryRouterTool()
    tools = [search_tool, web_content_tool, query_router_tool]

    # Updated prefix with tool_names
    prefix = """You are an advanced AI research assistant. Available tools: {tool_names}. Use them to help answer the user's question:

{tools}

Follow these steps:
1. First use query_router to classify the query
2. Choose appropriate tool(s) based on classification
3. Format response with verified sources

Response format:
- Concise direct answer
- Supporting bullet points
- Sources in [brackets]"""

    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", prefix),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Updated memory initialization
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create agent with the correct format
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Create executor with better error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=4,
        early_stopping_method="generate"
    )

    return agent_executor

# Improved process_query function with error handling
def process_query(agent_executor, query: str) -> str:
    try:
        # Add timeout handling
        start_time = time.time()
        timeout = 60  # 60 seconds timeout
        
        result = agent_executor.invoke({"input": query})  # Changed from () to .invoke()
        
        # Check if processing took too long
        if time.time() - start_time > timeout:
            return "Query processing timed out. Please try a simpler query or break it into smaller parts."
            
        return result["output"]
    except KeyError as e:
        return f"Error in response format: {str(e)}"
    except Exception as e:
        error_message = str(e)
        print(f"Error details: {error_message}")
        return f"Error processing query: {error_message}"

def main():
    print("Initializing Personal AI Research Agent with Llama 3.2...")
    try:
        # Check if Ollama is available
        try:
            llm = initialize_llm()
            # Test the LLM with a simple query
            test_result = llm.invoke("Test")
            print("LLM connection successful!")
        except Exception as e:
            print(f"Error connecting to Ollama: {str(e)}")
            print("Please make sure Ollama is installed and running with the Llama 3.2 model.")
            return
            
        agent_executor = create_research_agent()
        print("Agent initialized successfully! Type 'exit' to quit.")
        
        while True:
            query = input("\nEnter your query: ")
            
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            
            print("\nProcessing query...")
            start_time = time.time()
            response = process_query(agent_executor, query)
            end_time = time.time()
            
            print(f"\nResponse (took {end_time - start_time:.2f} seconds):")
            print(response)
    except Exception as e:
        print(f"Error initializing agent: {str(e)}")
        print("Check your Ollama installation and ensure Llama 3.2 model is available.")

def create_web_interface():
    from flask import Flask, request, jsonify, render_template_string
    
    app = Flask(__name__)
    agent_executor = None  # Initialize later to avoid slow startup
    
    @app.route('/')
    def home():
        html = r'''
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Research Assistant</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                #chat-container { height: 500px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; }
                .message { margin-bottom: 10px; padding: 8px; border-radius: 5px; }
                .user { background-color: #f0f0f0; text-align: right; }
                .assistant { background-color: #e1f5fe; }
                #input-container { display: flex; }
                #query-input { flex-grow: 1; padding: 10px; }
                button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
                .loading { text-align: center; margin: 10px 0; }
                pre { white-space: pre-wrap; word-wrap: break-word; }
                .sources { font-style: italic; color: #666; }
            </style>
        </head>
        <body>
            <h1>AI Research Assistant</h1>
            <div id="chat-container"></div>
            <div id="input-container">
                <input type="text" id="query-input" placeholder="Enter your query...">
                <button onclick="sendQuery()" id="send-button">Send</button>
            </div>
            
            <script>
                function sendQuery() {
                    const queryInput = document.getElementById('query-input');
                    const query = queryInput.value;
                    const sendButton = document.getElementById('send-button');
                    
                    if (!query) return;
                    
                    addMessage('user', query);
                    queryInput.value = '';
                    
                    // Disable button during processing
                    sendButton.disabled = true;
                    sendButton.textContent = 'Processing...';
                    
                    // Show loading indicator
                    const loadingId = showLoading();
                    
                    fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        // Remove loading indicator
                        hideLoading(loadingId);
                        
                        // Format the response to highlight sources
                        const formattedResponse = formatResponse(data.response);
                        addMessage('assistant', formattedResponse);
                        
                        // Re-enable button
                        sendButton.disabled = false;
                        sendButton.textContent = 'Send';
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        // Remove loading indicator
                        hideLoading(loadingId);
                        addMessage('assistant', 'Error processing your query.');
                        
                        // Re-enable button
                        sendButton.disabled = false;
                        sendButton.textContent = 'Send';
                    });
                }
                
                function formatResponse(text) {
                    // Highlight sources in [brackets]
                    return text.replace(/\[([^\]]+)\]/g, '<span class="sources">[$1]</span>');
                }
                
                function addMessage(sender, text) {
                    const chatContainer = document.getElementById('chat-container');
                    const messageElement = document.createElement('div');
                    messageElement.className = `message ${sender}`;
                    messageElement.innerHTML = `<strong>${sender === 'user' ? 'You' : 'Assistant'}:</strong> ${text}`;
                    chatContainer.appendChild(messageElement);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
                
                function showLoading() {
                    const chatContainer = document.getElementById('chat-container');
                    const loadingElement = document.createElement('div');
                    loadingElement.className = 'loading';
                    loadingElement.innerHTML = 'Processing query...';
                    loadingElement.id = 'loading-' + Date.now();
                    chatContainer.appendChild(loadingElement);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    return loadingElement.id;
                }
                
                function hideLoading(id) {
                    const loadingElement = document.getElementById(id);
                    if (loadingElement) {
                        loadingElement.remove();
                    }
                }
                
                // Allow sending with Enter key
                document.getElementById('query-input').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendQuery();
                    }
                });
            </script>
        </body>
        </html>
        '''
        return render_template_string(html)
    
    @app.route('/query', methods=['POST'])
    def query():
        global agent_executor
        
        # Initialize agent_executor if not already done
        if agent_executor is None:
            try:
                agent_executor = create_research_agent()
            except Exception as e:
                return jsonify({'response': f"Error initializing agent: {str(e)}"})
        
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'response': 'No query provided'}), 400
        
        response = process_query(agent_executor, query)
        return jsonify({'response': response})
    
    return app

if __name__ == "__main__":
    main()