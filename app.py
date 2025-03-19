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
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="SkillSync AI",
                   layout="wide",
                   page_icon='logo.jpg')

st.title("Find Your Dream Job!")

class SkillGaps(BaseModel):
    profile_summary: str = Field(description="Extract the profile summary like skills, projects, education, experience")
    strengths: list[str] = Field(description="Extract all the strenghts of the candidate.")
    weaknesses: list[str] = Field(description="Extract the weaknesses and skill gaps of the candidate.")
    areas_of_improvement: list[str] = Field(description="Extract the areas of improvement for the candidate in question.")


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

if "agent" not in st.session_state:
    st.session_state.agent=create_web_crawler_and_study_materials_agent()

btn1=st.sidebar.button("Submit")

if btn1:
    career_summary=get_resume_details(text=text, model='gemini')
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

    job_roles=career_summary.job_role
    
    industry_trends=st.session_state.agent.invoke({"messages":f"Do a detailed search for the required skillsets and current trend for the job roles {job_roles} and provide your answers in a clean format."})

    industry_trends_results=str(industry_trends.get("messages")[-2].content)+str(industry_trends.get("messages")[-1].content)

    agent_prompt_template=PromptTemplate(
        template="""
            You will be given a career summary of the candidate which is as follows: \n{career_summary}.\n You will also be given required skillsets and industry trends for one or more than one job roles {job_roles} which is as follows which will be given to you: {industry_trends_results}. \n You need to compare the candidates profile with the insdustry trends that will be given to you and find out the skill gaps, strengths, weaknesses, areas of improvement.
        """,
        input_variables=["career_summary","job_roles","industry_trends_results"]
    )

    skill_gaps_chain=agent_prompt_template|model
    skill_gaps=skill_gaps_chain.invoke({
        "career_summary":career_summary,
        "job_roles":job_roles,
        "industry_trends_results":industry_trends_results
    })

    st.markdown(skill_gaps.content)