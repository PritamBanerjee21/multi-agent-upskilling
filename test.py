from helper_functions import load_env_variables
import os

from helper_functions import load_file, get_suggestions
from agents import create_web_crawler_and_study_materials_agent

# a,b,c=load_env_variables()

# print(f"Gemini: {a}\nYouTube: {b}\nGROQ: {c}")
# print(f"OpenAI: {os.environ["OPENAI_API_KEY"]}")
# print(f"TAVILY: {os.environ["TAVILY_API_KEY"]}")

# file_path=r"D:\Job Documents\Updated Resume (Data Scientist).pdf"
file_path="resume_ss.png"
# print(file_path)

gemini_api_key,youtube_api_key,groq_api_key=load_env_variables()
agent=create_web_crawler_and_study_materials_agent(model="gemini")

text=load_file(file_path)
results=get_suggestions(text=text,agent=agent,api_keys={"GEMINI_API_KEY":gemini_api_key,
                                                        "YOUTUBE_API_KEY":youtube_api_key,
                                                        "GROQ_API_KEY":groq_api_key})

print(results)