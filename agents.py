"""
Agent creation and management module.
This module contains functions for creating and configuring AI agents for resume analysis,
web crawling, and study materials generation.
"""

# from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.tools import WikipediaQueryRun, YouTubeSearchTool, DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_community.tools.openai_dalle_image_generation import OpenAIDALLEImageGenerationTool
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory

from langgraph.prebuilt import create_react_agent
from googleapiclient.discovery import build
from langgraph.checkpoint.sqlite import SqliteSaver

import os
from dotenv import load_dotenv
import requests

load_dotenv()

# Load API keys from environment variables
youtube_api_key=os.getenv("YOUTUBE_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")

# Initialize Gemini model
model=ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=gemini_api_key
)

def fetch_coursera_courses(query: str):
    """
    Fetch courses from Coursera API based on a search query.
    
    Args:
        query (str): Search query for courses
        
    Returns:
        str: Formatted course information including names, descriptions, and URLs
    """
    url = f"https://api.coursera.org/api/courses.v1?q=search&query={query}&fields=name,description,partnerIds,slug"

    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        courses = data.get("elements", [])

        results = []
        for course in courses[:5]:  # Limit to top 5 results
            name = course.get("name", "Unknown Course")
            desc = course.get("description", "No description available")
            slug = course.get("slug", "")
            course_url = f"https://www.coursera.org/learn/{slug}" if slug else "URL not available"

            results.append(f"**Name: {name}**\nDescription: {desc}\n🔗 [Course Link]({course_url})")

        return "\n\n".join(results) if results else "No courses found."

    else:
        return f"Error: {response.status_code} - {response.text}"

def youtube_search(query, max_results=5):
    """
    Search YouTube videos using the YouTube API.
    
    Args:
        query (str): Search query for videos
        max_results (int): Maximum number of results to return
        
    Returns:
        str: Formatted video information including titles, descriptions, and URLs
    """
    youtube = build("youtube", "v3", developerKey=youtube_api_key)

    search_response = youtube.search().list(
        q=query,
        part="snippet",
        maxResults=max_results,
        type="video"
    ).execute()

    video_data = []

    for video in search_response.get("items", []):
        video_id = video['id']['videoId']
        title = video['snippet']['title']
        description = video['snippet']['description']
        channel_title = video['snippet']['channelTitle']
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        video_data.append(
            f"**Title: {title}**\n📺 Channel: {channel_title}\n📝 Description: {description}\n🔗 [Watch Here]({video_url})"
        )

    return "\n\n".join(video_data) if video_data else "No videos found."

# def custom_state_modifier(state):
#     """
#     Allows the agent to use tools, but if no tool is applicable,
#     it generates a response using its own knowledge.
#     """
#     if "tool" not in state:  # If no tool is chosen
#         state["output"] = model.invoke(state["messages"])  # Let the model generate response
#     return state

def custom_state_modifier(state):
    """
    Modifies the state to either use tools or generate a response from the model.
    
    Args:
        state (dict): Current state of the agent
        
    Returns:
        dict: Modified state
    """
    if "messages" not in state:
        state["messages"] = []  # Ensure messages list exists

    if "tool" not in state:  # If no tool is chosen, generate response
        model_response = model.invoke(state["messages"])
        state["messages"].append(AIMessage(content=model_response.content))  # Append AI response

    return state

def generate_text(query):
    """
    Generate text response from the model's knowledge base.
    
    Args:
        query: Input query for the model
        
    Returns:
        str: Generated text response
    """
    return model.invoke(query).content  # Assuming `model` is your language model

def create_model(name):
    """
    Create and configure a language model based on the specified name.
    
    Args:
        name (str): Model name ("groq", "gpt", or "gemini")
        
    Returns:
        ChatModel: Configured language model
        str: Error message if invalid model name
    """
    if name=="groq":
        model=ChatGroq(
            model="qwen-2.5-32b",
            api_key=groq_api_key
        )
        return model

    elif name=='gpt' or name=='openai':
        model=ChatOpenAI(
            model='gpt-4o'
        )
        return model

    elif name=='gemini':
        model=ChatGoogleGenerativeAI(
            model='gemini-2.0-flash',
            api_key=os.getenv("GEMINI_API_KEY")
        )
        return model

    else:
        return "please provide one among the following - 'gpt', 'groq', 'gemini'"
    

def create_web_crawler_and_study_materials_agent(model="groq"):
    """
    Create an agent for web crawling and study materials generation.
    
    Args:
        model (str): Base model to use ("groq", "gemini", or "gpt")
        
    Returns:
        Agent: Configured agent with tools and capabilities
    """
    if model=="groq":
        model=ChatGroq(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            api_key=groq_api_key
        )

    elif model=="gemini":
        model=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY")
        )

    elif model=="gpt" or model=="openai":
        model=ChatOpenAI(
            model='gpt-4o'
        )

    # Initialize Wikipedia tool
    wiki_api_wrapper=WikipediaAPIWrapper(top_k_results=5)
    wiki_tool=WikipediaQueryRun(api_wrapper=wiki_api_wrapper,
                                description="A tool to explain things in text format. Use this tool if you think the user's asked concept is best explained through text.",
                                name="wiki_tool")

    # Initialize DuckDuckGo search tool
    duckduckgo_api_wrapper=DuckDuckGoSearchAPIWrapper()
    duckduckgo_search_tool=DuckDuckGoSearchResults(api_wrapper=duckduckgo_api_wrapper,
                            description="A search engine. Use this tool if you need to answer questions about current events. Input should be a search query.",
                            name="duckduckgo_search_tool")

    # Initialize YouTube search tool
    youtube_search_tool=Tool(
        name="YouTube_Video_Search",
        func=youtube_search,
        description="Searches for YouTube videos related to a given query and returns the top 5 videos along with their links, titles, descriptions, and channel names"
    )

    # Initialize Coursera search tool
    coursera_tool=Tool(
        name="coursera_search_tool",
        func=fetch_coursera_courses,
        description="Fetch courses on the given topic."
    )

    system_prompt="You are a helpful assistant named Friday."

    # Initialize text generation tool
    text_gen_tool = Tool(
        name="text_generator",
        func=generate_text,
        description="Use this tool to generate essays, summaries, or general text responses."
    )

    # Combine all tools
    tools=[wiki_tool,coursera_tool,duckduckgo_search_tool,youtube_search_tool,text_gen_tool]
    # memory=SqliteSaver("agent_memory.sqlite")
    # store = SQLiteStore(database_path="memory.sqlite")
    # memory=ConversationBufferMemory(memory_key="chat_history",chat_memory=store)

    agent = create_react_agent(
        model=model,  
        tools=tools,  # Keep the tools for structured outputs
        state_modifier=system_prompt  # Allow direct responses
        # checkpointer=memory
    )

    return agent