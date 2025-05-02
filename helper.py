"""
Helper functions for resume analysis and processing.
This module contains functions and classes for extracting and analyzing resume data,
including candidate details, skill gaps, and resume validation.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import google.generativeai as genai

import os

from typing import List, Literal, Optional
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

class CandidateDetails(BaseModel):
    """
    Pydantic model for storing candidate details extracted from resume.
    
    Attributes:
        skills (str): All skills of the candidate
        experience (str): Years of experience
        education (str): Education details (degree, stream, college)
        projects (List[str]): List of project summaries
        job_role (str): Recommended job role(s)
        location (Optional[str]): Candidate's location if mentioned
    """
    skills: str = Field(description="Extract all the skills of the candidate from the resume")
    experience: str = Field(description="Extract the number of years of experience of the candidate from the resume")
    education: str = Field(description="Extract the education details (like degree, stream, college, etc.) of the candidate from the resume")
    projects: list[str] = Field(description="Extract all the projects and put it inside a list. It should only contain a one liner gist of the project.")
    job_role: str = Field(description="Extract one or more job roles for the candidate according to the resume")
    location: Optional[str] = Field(description="Extract the location of the candidate if mentioned in the resume.")

class IsResume(BaseModel):
    """
    Pydantic model for validating if a document is a resume.
    
    Attributes:
        is_resume (Literal["yes", "no"]): Whether the document is a resume
    """
    is_resume: Literal["yes", "no"] = Field(description="Determine if the given text is a resume or not. If it is a resume, return \"yes\", else return \"no\"")

class SkillGaps(BaseModel):
    """
    Pydantic model for storing skill gap analysis results.
    
    Attributes:
        profile_summary (str): Summary of candidate's profile
        strengths (List[str]): List of candidate's strengths
        weaknesses (List[str]): List of candidate's weaknesses and skill gaps
        areas_of_improvement (List[str]): Areas where candidate can improve
    """
    profile_summary: str = Field(description="Extract the profile summary like skills, projects, education, experience")
    strengths: list[str] = Field(description="Extract all the strenghts of the candidate.")
    weaknesses: list[str] = Field(description="Extract the weaknesses and skill gaps of the candidate.")
    areas_of_improvement: list[str] = Field(description="Extract the areas of improvement for the candidate in question.")

# Load API keys from environment variables
groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["OPENAI_API_KEY"]=os.getenv('OPENAI_API_KEY')
gemini_api_key=os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

# Initialize Groq model
groq_model=ChatGroq(
    model='llama-3.3-70b-specdec',
    api_key=groq_api_key
)

def get_resume_details(text, model="groq"):
    """
    Extract and analyze details from a resume text.
    
    Args:
        text (str): The resume text to analyze
        model (str): The model to use for analysis ("groq", "gpt", or "gemini")
        
    Returns:
        CandidateDetails: Extracted candidate details
        str: Error message if document is not a resume
    """
    if model=="groq":
        model_=ChatGroq(
            model='llama-3.3-70b-versatile',
            api_key=groq_api_key
        )
    
    elif model=='gpt' or model=="openai":
        model_=ChatOpenAI(
            model="gpt-4o"
        )

    elif model=='gemini':
        model_=ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=gemini_api_key
        )

    # Initialize resume validation parser
    parser_is_resume=PydanticOutputParser(pydantic_object=IsResume)

    # Create prompt for resume validation
    prompt1=PromptTemplate(
        template="You'll receive a text below. You have to figure out whether it is a proper resume or not. \n {text} \n {format_instructions}",
        input_variables=["text"],
        partial_variables={
            "format_instructions":parser_is_resume.get_format_instructions()
        }
    )

    # Initialize candidate details parser
    parser_candidate_details=PydanticOutputParser(pydantic_object=CandidateDetails)

    # Create prompt for extracting candidate details
    prompt2=PromptTemplate(
        template="You'll receive a resume below. You have to extract the essential details from the resume: \n {text} \n {format_instructions}",
        input_variables=["text"],
        partial_variables={
            "format_instructions":parser_candidate_details.get_format_instructions()
        }
    )

    # Validate if document is a resume
    chain1 = prompt1 | model_ | parser_is_resume
    response_is_resume=chain1.invoke({"text":text})

    if response_is_resume.is_resume=="no":
        return "The document you uploaded is not a resume. Please enter a valid document."

    else:
        # Extract candidate details
        chain2 = prompt2 | model_ | parser_candidate_details
        return chain2.invoke({"text":text})

def get_resume_details_from_image(image_file_path):
    """
    Extract text from a resume image using Gemini model.
    
    Args:
        image_file_path: Path to the resume image file
        
    Returns:
        str: Extracted text from the resume image
    """
    text_extractor=genai.GenerativeModel("gemini-2.0-flash")
    image=Image.open(image_file_path)

    response=text_extractor.generate_content(["Extract all the details from the given resume.",image])

    resume_details=response.to_dict()['candidates'][0]["content"]["parts"][0]["text"]

    return resume_details