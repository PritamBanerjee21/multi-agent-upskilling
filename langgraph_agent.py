"""
LangGraph agent implementation for resume analysis and career guidance.
This module provides functionality for creating and managing LangGraph agents
that can analyze resumes and provide personalized career recommendations.
"""

from typing import Annotated

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

import os
from dotenv import load_dotenv

load_dotenv()

# Load API keys from environment variables
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Initialize embedding model
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers-embedding")

class State(TypedDict):
    """
    TypedDict for representing the state of the LangGraph agent.
    
    Attributes:
        messages (list): List of messages in the conversation
    """
    messages: Annotated[list, add_messages]

def load_and_prepare_file(file_path, return_text, chunk_size=200, chunk_overlap=10):
    """
    Load and prepare a file for vector storage and analysis.
    
    Args:
        file_path (str): Path to the file to load
        return_text (str): Text content to include in vector store
        chunk_size (int): Size of text chunks for processing
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        FAISS: Vector store containing the processed document
    """
    # Generate a unique path for this session's vectorstore
    file_name = os.path.basename(file_path)
    session_id = os.path.splitext(file_name)[0]
    vectorstore_path = f"cached_faiss_store_{session_id}"
    
    # Always create a new vectorstore for each resume to ensure fresh data
    print(f"Creating vectorstore for {file_name}...")
    
    # Handle different file types
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=file_path)
        docs = loader.load()
    elif file_extension in ['.doc', '.docx']:
        try:
            # First try UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
        except Exception as e:
            print(f"UnstructuredWordDocumentLoader failed: {e}, trying Docx2txtLoader...")
            try:
                # Fallback to Docx2txtLoader
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            except Exception as e2:
                print(f"Docx2txtLoader failed: {e2}, using extracted text directly...")
                # If both loaders fail, use the extracted text directly
                docs = [Document(page_content=return_text, metadata={"source": file_path})]
    else:
        # For non-PDF/Word files, create a document from the extracted text
        docs = [Document(page_content=return_text, metadata={"source": file_path})]
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the documents into chunks
    if len(docs) > 0:
        docs = splitter.split_documents(docs)
    
    # Create vector store from documents
    vector_store = FAISS.from_documents(docs, embedding_model)
    
    # Add the extracted text as additional context
    if return_text:
        text_chunks = splitter.split_text(return_text)
        for chunk in text_chunks:
            vector_store.add_documents([Document(page_content=chunk, metadata={"source": "extracted_text"})])
    
    return vector_store

def create_langgraph_agent(model: str, file_path: str, return_text: str, chunk_size=200, chunk_overlap=20):
    """
    Create a LangGraph agent for resume analysis.
    
    Args:
        model (str): Model to use ("gemini", "groq", or "openai")
        file_path (str): Path to the resume file
        return_text (str): Text content to include in analysis
        chunk_size (int): Size of text chunks for processing
        chunk_overlap (int): Overlap between chunks
        
    Returns:
        tuple: (Graph, VectorStore) - The configured agent and its vector store
    """
    graph_builder = StateGraph(State)

    memory=MemorySaver()

    # Initialize search tool and vector store
    search_tool = TavilySearchResults(max_results=5)
    vector_store = load_and_prepare_file(
        file_path=file_path,
        return_text=return_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Initialize language model based on specified type
    if model=="gemini":
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=gemini_api_key
        )
    elif model=="groq":
        llm=ChatGroq(
            model="qwen-qwq-32b",
            api_key=groq_api_key
        )

    elif model=="openai":
        llm=ChatOpenAI(
            model="gpt-4o"
        )

    else:
        raise ValueError("Please provide one among the following: 'gemini', 'groq', 'openai'")

    @tool
    def recall_memory(query: str) -> str:
        """
        Tool to answer questions about the user's career, skills, and resume.
        
        Args:
            query (str): Question about the resume or career
            
        Returns:
            str: Response based on resume information
        """
        try:
            # Perform similarity search directly on the vector store instance
            results = vector_store.similarity_search(query, k=3)
            
            if not results:
                return "I couldn't find relevant information in your resume. Could you please rephrase your question or be more specific?"
            
            memory_context = "\n\n".join([r.page_content for r in results])
            
            model_input = f"""Based on the following resume information, answer the question: {query}

RESUME CONTEXT:
{memory_context}

IMPORTANT INSTRUCTIONS:
1. If the resume context contains relevant information to answer the question, use it to provide a specific, personalized response.
2. If the resume context doesn't contain enough information to answer the question, acknowledge this limitation and ask for more details.
3. Always respond as if you're speaking directly to the resume owner.
4. For questions about skill gaps, compare the skills in the resume with industry standards.
5. For career advice, base your recommendations on the specific experience and skills mentioned in the resume."""
            
            response = llm.invoke(model_input)
            return response.content
        except Exception as e:
            print(f"Error in recall_memory: {str(e)}")
            return "I encountered an error while accessing your resume information. Could you please rephrase your question?"

    # Configure tools and agent
    tools = [search_tool, recall_memory]
    llm_with_tools = llm.bind_tools(tools)
    
    def chatbot(state: State):
        """
        Chatbot function that processes messages and generates responses.
        
        Args:
            state (State): Current state of the conversation
            
        Returns:
            dict: Updated state with new message
        """
        message = llm_with_tools.invoke(state["messages"])
        # Because we will be interrupting during tool execution,
        # we disable parallel tool calling to avoid repeating any
        # tool invocations when we resume.
        assert len(message.tool_calls) <= 1
        return {"messages": [message]}
    
    # Build the graph
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    graph = graph_builder.compile(checkpointer=memory)
    return graph, vector_store

def get_response(user_input:str, config:dict, agent):
    """
    Get a response from the agent for a given input.
    
    Args:
        user_input (str): User's question or input
        config (dict): Configuration for the agent
        agent: The LangGraph agent instance
        
    Returns:
        str: Agent's response to the input
    """
    try:
        outputs = []
        events = agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    },
                ],
            },
            config=config,  # Pass config directly
            stream_mode="values",
        )
        for event in events:
            if "messages" in event:
                outputs.extend(event["messages"])
        
        if not outputs:
            return "I apologize, but I couldn't generate a response. Please try asking your question differently."
            
        result = outputs[-1].content if outputs else ""
        return result
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return "I encountered an error while processing your request. Please try again."