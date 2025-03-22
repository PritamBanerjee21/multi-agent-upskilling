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

    skills: str = Field(description="Extract all the skills of the candidate from the resume")
    experience: str = Field(description="Extract the number of years of experience of the candidate from the resume")
    education: str = Field(description="Extract the education details (like degree, stream, college, etc.) of the candidate from the resume")
    projects: list[str] = Field(description="Extract all the projects and put it inside a list. It should only contain a one liner gist of the project.")
    job_role: str = Field(description="Extract one or more job roles for the candidate according to the resume")
    location: Optional[str] = Field(description="Extract the location of the candidate if mentioned in the resume.")

class IsResume(BaseModel):
    is_resume: Literal["yes", "no"] = Field(description="Determine if the given text is a resume or not. If it is a resume, return \"yes\", else return \"no\"")

class SkillGaps(BaseModel):
    profile_summary: str = Field(description="Extract the profile summary like skills, projects, education, experience")
    strengths: list[str] = Field(description="Extract all the strenghts of the candidate.")
    weaknesses: list[str] = Field(description="Extract the weaknesses and skill gaps of the candidate.")
    areas_of_improvement: list[str] = Field(description="Extract the areas of improvement for the candidate in question.")

groq_api_key=os.getenv('GROQ_API_KEY')
os.environ["OPENAI_API_KEY"]=os.getenv('OPENAI_API_KEY')
gemini_api_key=os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

groq_model=ChatGroq(
    model='llama-3.3-70b-specdec',
    api_key=groq_api_key
)

def get_resume_details(text, model="groq"):

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

    parser_is_resume=PydanticOutputParser(pydantic_object=IsResume)

    prompt1=PromptTemplate(
        template="You'll receive a text below. You have to figure out whether it is a proper resume or not. \n {text} \n {format_instructions}",
        input_variables=["text"],
        partial_variables={
            "format_instructions":parser_is_resume.get_format_instructions()
        }
    )

    parser_candidate_details=PydanticOutputParser(pydantic_object=CandidateDetails)

    prompt2=PromptTemplate(
        template="You'll receive a resume below. You have to extract the essential details from the resume: \n {text} \n {format_instructions}",
        input_variables=["text"],
        partial_variables={
            "format_instructions":parser_candidate_details.get_format_instructions()
        }
    )

    chain1 = prompt1 | model_ | parser_is_resume

    response_is_resume=chain1.invoke({"text":text})

    if response_is_resume.is_resume=="no":
        return "The document you uploaded is not a resume. Please enter a valid document."

    else:
        chain2 = prompt2 | model_ | parser_candidate_details
        return chain2.invoke({"text":text})

def get_resume_details_from_image(image_file_path):
    
    text_extractor=genai.GenerativeModel("gemini-2.0-flash")
    image=Image.open(image_file_path)

    response=text_extractor.generate_content(["Extract all the details from the given resume.",image])

    resume_details=response.to_dict()['candidates'][0]["content"]["parts"][0]["text"]

    return resume_details
    # parser_resume_details=PydanticOutputParser(pydantic_object=CandidateDetails)

    # prompt=PromptTemplate(
    #     template="""
    #         You will get resume details as follows: \n {resume_details}. You have to parse it as follows: \n {format_instructions}
    #     """,
    #     input_variables=["resume_details"],
    #     partial_variables={
    #         "format_instructions":parser_resume_details.get_format_instructions()
    #     }
    # )

    # if model=="groq":
    #     model=ChatGroq(
    #         model="qwen-2.5-32b",
    #         api_key=groq_api_key
    #     )

    # elif model=="gemini":
    #     model=ChatGoogleGenerativeAI(
    #         model="gemini-2.0-flash",
    #         api_key=gemini_api_key
    #     )

    # # parser=PydanticOutputParser(pydantic_object=CandidateDetails)
    # image_resume_chain=prompt|model|parser_resume_details

    # result=image_resume_chain.invoke({
    #     "resume_details":resume_details
    # })

    # return result