import streamlit as st
import os
from dotenv import load_dotenv
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document
from helper import CandidateDetails, IsResume, get_resume_details
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

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="SkillSync AI",
                   layout="wide",
                   page_icon='logo.jpg')

st.title("Find Your Dream Job!")

with st.sidebar:
    uploaded_file=st.file_uploader("Upload your resume:",type=["pdf","docx","doc"])

    if uploaded_file is not None:

        file_extension = uploaded_file.name.split('.')[-1]
        file_buffer = BytesIO(uploaded_file.getbuffer())

        if file_extension == "pdf":
            reader = PdfReader(file_buffer)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        elif file_extension in ["docx", "doc"]:
            doc = Document(file_buffer)  # Load DOCX directly from memory
            text = "\n".join([para.text for para in doc.paragraphs])

        else:
            st.error("Unsupported file format.")

    else:
        st.warning("Please enter a file!")

btn1=st.sidebar.button("Submit")

if btn1:
    response=get_resume_details(text=text, model='gemini')
    prompt_extract_details=PromptTemplate(
        template="""You will be given academic details about a person which will be extracted from their resume beforehand. I need you to give me a summary of the candidate. The details are as follows:
            {details}
        """,
        input_variables=["details"]
    )
    model=ChatGoogleGenerativeAI(
        model='gemini-2.0-flash',
        api_key=os.getenv("GEMINI_API_KEY")
    )
    # prompt=prompt_extract_details.invoke({"details":response})
    chain=prompt_extract_details|model
    response2=chain.invoke({"details":response})
    st.markdown(response2.content)