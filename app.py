"""
Flask application for SkillSync AI - A career guidance and resume analysis platform.
This application provides features for resume analysis, ATS scoring, and personalized career recommendations.
"""

from flask import (
    Flask,
    render_template,
    session,
    redirect,
    url_for,
    request,
    flash,
    Response,
    stream_with_context,
    jsonify
)
import os
import json
import uuid
from werkzeug.utils import secure_filename
from helper_functions import load_file, get_suggestions, load_env_variables
from agents import create_web_crawler_and_study_materials_agent
from ats_score import score_resume
from dotenv import load_dotenv
import time
from memory import get_or_create_agent, clear_agent
from langgraph_agent import get_response

# Load environment variables
load_dotenv()

# Load API keys
gemini_api_key, youtube_api_key, groq_api_key = load_env_variables()

# Initialize Flask application
app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'jpg', 'jpeg', 'png'}

# Set Flask configuration
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["SECRET_KEY"] = "visca_el_barca"

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    
    Args:
        filename (str): Name of the file to check
        
    Returns:
        bool: True if file extension is allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Create a global agent instance for resume analysis
agent = create_web_crawler_and_study_materials_agent(model="gemini")

@app.route("/")
@app.route("/home")
def home():
    """
    Render the home page.
    
    Returns:
        str: Rendered home.html template
    """
    return render_template("home.html")

@app.route("/explore")
def explore():
    """
    Render the explore page.
    
    Returns:
        str: Rendered explore.html template
    """
    return render_template("explore.html")

@app.route("/results")
def results():
    """
    Render the results page with extracted resume information.
    
    Returns:
        str: Rendered results.html template with resume data
    """
    # Get the extracted text from the session
    extracted_text = session.get('extracted_text', '')
    filename = session.get('filename', '')
    file_type = session.get('file_type', '')
    
    # Generate a unique session ID if not already present
    if 'chat_session_id' not in session:
        session['chat_session_id'] = str(uuid.uuid4())
    
    return render_template("results.html", 
                         extracted_text=extracted_text,
                         filename=filename,
                         file_type=file_type)

@app.route("/stream_suggestions")
def stream_suggestions():
    """
    Stream career suggestions based on the uploaded resume.
    Uses Server-Sent Events (SSE) to provide real-time updates.
    
    Returns:
        Response: SSE response with career suggestions
    """
    def generate():
        extracted_text = session.get('extracted_text', '')
        if not extracted_text:
            yield "data: No text found\n\n"
            yield "event: close\ndata: Stream closed\n\n"
            return

        # Don't re-run if already generated
        if session.get('suggestions_generated', False):
            # Return a specific message that the client can recognize
            yield "data: \"Suggestions already generated\"\n\n"
            yield "event: close\ndata: Stream closed\n\n"
            return

        api_keys = {
            "GEMINI_API_KEY": gemini_api_key,
            "YOUTUBE_API_KEY": youtube_api_key,
            "GROQ_API_KEY": groq_api_key
        }

        try:
            # Set the flag before processing to prevent concurrent requests
            session['suggestions_generated'] = True
            session.modified = True  # Ensure the session is saved immediately
            
            suggestions = get_suggestions(extracted_text, agent, api_keys)

            # Get content from the last message
            if isinstance(suggestions, dict) and "messages" in suggestions and suggestions["messages"]:
                content = suggestions["messages"][-1].content
            else:
                content = str(suggestions)

            yield f"data: {json.dumps({'suggestions': content})}\n\n"
            yield "event: close\ndata: Stream closed\n\n"

        except Exception as e:
            # If there's an error, reset the flag so it can be tried again
            session['suggestions_generated'] = False
            session.modified = True
            
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "event: close\ndata: Stream closed\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route("/handle_upload", methods=["GET", "POST"])
def handle_upload():
    """
    Handle file upload and process the resume.
    
    Returns:
        Response: Redirect to results page or explore page
    """
    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)
    
    files = request.files.getlist("file")

    for file in files:
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Extract text from the file
            extracted_text = load_file(file_path)
            
            # Store the results in the session
            session['extracted_text'] = extracted_text
            session['filename'] = filename
            session['file_type'] = filename.split('.')[-1].upper()
            
            # Generate a new session ID for the chat
            session['chat_session_id'] = str(uuid.uuid4())
            
            # Clear any existing agent for this session
            if 'chat_session_id' in session:
                clear_agent(session['chat_session_id'])
            
            # Redirect to the results page immediately
            return redirect(url_for('results'))

    return redirect(url_for("explore"))

@app.route("/ats_score")
def ats_score():
    """
    Render the ATS score page.
    
    Returns:
        str: Rendered ats_score.html template
    """
    return render_template("ats_score.html")

@app.route("/calculate_ats_score", methods=["POST"])
def calculate_ats_score():
    """
    Calculate ATS score for the uploaded resume against a job description.
    
    Returns:
        Response: JSON response with ATS score and analysis
    """
    try:
        job_description = request.form.get('job_description')
        if not job_description:
            return jsonify({"error": "Job description is required"}), 400

        # Get the file path from the session
        filename = session.get('filename')
        if not filename:
            return jsonify({"error": "No resume file found. Please upload a resume first."}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "Resume file not found. Please upload your resume again."}), 400
        
        # Calculate ATS score
        try:
            result = score_resume(file_path, job_description, scoring_function="extract")
            
            # Convert sets to lists for JSON serialization
            result["matched_skills"] = list(result["matched_skills"])
            result["matched_keywords"] = list(result["matched_keywords"])
            result["important_skills"] = list(result["important_skills"])
            
            return jsonify(result)
        except Exception as e:
            app.logger.error(f"Error in score_resume: {str(e)}")
            return jsonify({"error": f"Error calculating score: {str(e)}"}), 500
            
    except Exception as e:
        app.logger.error(f"Error in calculate_ats_score: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/reset_suggestions", methods=["POST"])
def reset_suggestions():
    """
    Reset the suggestions flag to allow generating new suggestions.
    
    Returns:
        Response: JSON response indicating success
    """
    session['suggestions_generated'] = False
    return jsonify({"status": "success"})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chat interactions with the AI agent.
    
    Returns:
        Response: JSON response with AI's reply
    """
    try:
        message = request.json.get('message', '')
        if not message:
            return jsonify({"error": "No message provided"}), 400
            
        session_id = session.get('chat_session_id')
        if not session_id:
            session['chat_session_id'] = str(uuid.uuid4())
            session_id = session['chat_session_id']
            
        filename = session.get('filename')
        extracted_text = session.get('extracted_text', '')
        
        if not filename or not extracted_text:
            return jsonify({"error": "No resume found. Please upload a resume first."}), 400
            
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Get or create the agent for this session
        agent = get_or_create_agent(session_id, file_path, extracted_text)
        
        # Configure the agent with the session ID
        config = {
            "configurable": {
                "thread_id": session_id,
                "vector_store_enabled": True  # Add flag to indicate vector store availability
            }
        }
        
        # Get response from the agent
        response = get_response(message, config, agent)
        
        return jsonify({"response": response})
        
    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)