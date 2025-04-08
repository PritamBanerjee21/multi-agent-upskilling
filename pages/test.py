import streamlit as st
import os
from dotenv import load_dotenv
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from helper import CandidateDetails, IsResume, get_resume_details, SkillGaps, get_resume_details_from_image
from agents import create_web_crawler_and_study_materials_agent
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.document_loaders import TextLoader


from langgraph_agent import create_langgraph_agent, get_response

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")
gemini_api_key=os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="SkillSync AI",
                   layout="wide",
                   page_icon='logo.jpg')

st.title("Find Your Dream Job!")


if "langgraph_agent" not in st.session_state:
    st.session_state.langgraph_agent=create_langgraph_agent(model="gemini")

if "config" not in st.session_state:
    st.session_state.config={
        "configurable":{"thread_id":"1"}
    }

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

config={"configurable":{"thread_id":"1"}}

user_input = st.chat_input("Write your query:")

if user_input:

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    response=get_response(user_input=user_input,agent=st.session_state.langgraph_agent,config=st.session_state.config)
    st.session_state.chat_history.append({"role": "assistant", "content": response})


    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

            # if user_input is not None:
                # if user_input.strip() == "":
                #     st.error("Bro... at least type *something* 😑")
                # else:


    # response = get_response(user_input=user_input, config=config, agent=st.session_state.langgraph_agent)


    # for msg in st.session_state.chat_history:
    #     with st.chat_message(msg["role"]):
    #         st.markdown(msg["content"])