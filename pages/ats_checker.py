import streamlit as st
import os
from io import BytesIO
from ats_score import *

st.set_page_config(
    page_title="ATS Checker",
    layout="wide"
)

uploaded_file=st.file_uploader(
    "Upload your resume:",
    type=["pdf","docx"]
)

job_description=st.text_area("Enter the Job Description:")
submit_button=st.button("Submit")

if submit_button:

    if uploaded_file:

        file_path=uploaded_file.name
        file_buffer=BytesIO(uploaded_file.getbuffer())

        result=score_resume(resume_path=file_buffer,job_description=job_description)

        score_color = "#4CAF50"  

        if result["final_score"]>60:
            st.markdown(
                f"""
                <h2 style='text-align: center; color: white;'>
                    ATS Score for the given Job Description:
                    <span style='color: {score_color};'>{result['final_score']}</span>
                </h2>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                f"""
                <h2 style='text-align: center; color: white;'>
                    ATS Score for the given Job Description:
                    <span style='color: red;'>{result['final_score']}</span>
                </h2>
                """,
                unsafe_allow_html=True
            )

        matched_skills = result['matched_skills']

        # Render it on Streamlit
        st.markdown("<h3 style='color:white; text-align:center;'>✅ Matched Skills</h3>", unsafe_allow_html=True)

        for skill in matched_skills:

            st.markdown(f"""
            <div style='text-align: center; margin-top: 20px;'>
                <span style='
                    display: inline-block;
                    padding: 10px 22px;
                    border: 1px solid grey;  /* soft purple border */
                    border-radius: 30px;  /* nice visible pill shape */
                    color: #8A2BE2;
                    background-color: rgba(138, 43, 226, 0.1);  /* faint bg tint */
                    font-weight: 600;
                    font-size: 20px;
                    box-shadow: 0 4px 12px rgba(138, 43, 226, 0.2);
                    transition: all 0.3s ease-in-out;
                '>
                    {skill}
                </span>
            </div>
            """,unsafe_allow_html=True)

    else:
        st.warning("Please upload a file.")