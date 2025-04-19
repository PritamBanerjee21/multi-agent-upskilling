from langgraph.checkpoint.memory import MemorySaver
from langgraph_agent import create_langgraph_agent
import os

# Dictionary to store agent instances and their vector stores by session ID
agent_instances = {}
vector_stores = {}

def get_or_create_agent(session_id, file_path, return_text):
    """
    Get an existing agent instance or create a new one for the session.
    """
    if session_id in agent_instances:
        return agent_instances[session_id]
    
    # Create a new agent instance and store its vector store
    agent, vector_store = create_langgraph_agent(
        model="gemini",
        file_path=file_path,
        return_text=return_text,
        chunk_size=200,
        chunk_overlap=20
    )
    
    # Store both the agent instance and its vector store
    agent_instances[session_id] = agent
    vector_stores[session_id] = vector_store
    
    return agent

def get_vector_store(session_id):
    """
    Retrieve the FAISS vector store for the given session ID.
    """
    return vector_stores.get(session_id)

def clear_agent(session_id):
    """
    Remove an agent instance and its vector store from memory
    """
    if session_id in agent_instances:
        del agent_instances[session_id]
    if session_id in vector_stores:
        del vector_stores[session_id]
    return True