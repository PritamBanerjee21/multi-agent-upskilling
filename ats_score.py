#Importing libraries
import spacy
import fitz 
from sentence_transformers import SentenceTransformer, util
from spacy.matcher import PhraseMatcher
import re

# Extracting text
def extract_text_from_cv(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Preprocessing
def clean_text(jd):
    jd = re.sub(r'\n+', ' ', jd)  
    jd = re.sub(r'\s+', ' ', jd)  
    jd = re.sub(r'[^\w\s]', '', jd) 
    doc = nlp(jd.lower())  # lowercase and tokenize
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens, doc
    # return jd.lower()

# Extracting skills form jd
def extract_skills_from_jd(text, skill_list):
    tokens, doc = clean_text(text)
    
    # Normalize skill list
    normalized_skills = [skill.lower() for skill in skill_list]

    # Phrase matcher for multi-word skills
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in normalized_skills]
    matcher.add("SKILLS", patterns)

    matches = matcher(doc)
    extracted = set()

    for match_id, start, end in matches:
        span = doc[start:end]
        extracted.add(span.text.lower())

    return list(extracted)


# Lemmatizing
def get_lemmas(text):
    doc = nlp(text)
    return set([token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct])


# Extracting Keywords from JD
nlp = spacy.load("en_core_web_sm")

def extract_keywords(doc):
    # doc = nlp(jd)
    keywords = set()
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop:
            keywords.add(token.lemma_.lower())
    return keywords

# Keyword matching score
def keyword_match_score(resume_text, job_keywords):
    resume_lemmas = get_lemmas(resume_text)
    matched = resume_lemmas & job_keywords
    score = len(matched) / len(job_keywords) if job_keywords else 0
    return score * 100, matched


# Semantic similarity score
model = SentenceTransformer('all-MiniLM-L6-v2')  

def semantic_similarity(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return float(similarity[0][0]) * 100


def skill_match_score(resume_text, skill_list):
    resume_text_lower = resume_text.lower()
    matched_skills = [skill for skill in skill_list if skill.lower() in resume_text_lower]

    if len(matched_skills) >= 5:
        return 100.0, matched_skills
    elif len(matched_skills) == 4:
        return 90.0, matched_skills
    elif len(matched_skills) == 3:
        return 80.0, matched_skills
    elif skill_list:
        score = len(matched_skills) / len(skill_list)
        return score * 100, matched_skills
    else:
        return 0.0, []


# Final scoring
def calculate_final_score(keyword_score, semantic_score, skill_score):
    return (0.05 * keyword_score) + (0.50 * semantic_score) + (0.45 * skill_score)


# Complete scoring process
def score_resume(resume_path, job_description):
    resume_text = extract_text_from_cv(resume_path)
    jd_token, jd_doc = clean_text(job_description)

    job_keywords = extract_keywords(jd_doc)

    kw_score, matched_keywords = keyword_match_score(resume_text, job_keywords)
    sem_score = semantic_similarity(resume_text, job_description)
    sem_score = min(sem_score * 1.2, 100)
    
    all_skills = ["python", "java", "c++", "c#", "javascript", "typescript", "html", "css", "react", "angular", "vue.js", "node.js", "express.js", "sql", "nosql", "mongodb", "mysql", "postgresql", "oracle", "firebase", "aws", "azure", "gcp", "cloudformation", "terraform", "docker", "kubernetes", "linux", "bash", "powershell", "git", "github", "gitlab", "bitbucket", "devops", "ci/cd", "jenkins", "circleci", "agile", "scrum", "kanban", "jira", "confluence", "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "nlp", "computer vision", "data science", "data engineering", "data analysis", "data mining", "feature engineering", "pandas", "numpy", "scipy", "scikit-learn", "matplotlib", "seaborn", "plotly", "bokeh", "big data", "hadoop", "spark", "hive", "pig", "tableau", "power bi", "looker", "qlikview", "superset", "data visualization", "data wrangling", "data cleaning", "data preprocessing", "sql tuning", "database optimization", "etl", "elt", "airflow", "dbt", "snowflake", "redshift", "databricks", "azure data factory", "google dataflow", "cybersecurity", "networking", "penetration testing", "ethical hacking", "firewalls", "vpn", "ssl", "sso", "oauth", "blockchain", "web3", "iot", "cloud computing", "virtualization", "vmware", "hyper-v", "system administration", "database administration", "api development", "rest", "graphql", "grpc", "microservices", "monolith", "software testing", "unit testing", "integration testing", "automation testing", "selenium", "robot framework", "cypress", "playwright", "junit", "pytest", "mocking", "itil", "erp", "sap", "crm", "salesforce", "service now", "jupyter", "notebooks", "excel", "statistics", "probability", "hypothesis testing", "regression", "classification", "clustering", "time series", "model evaluation", "mle", "mlops", "version control", "prompt engineering", "llms", "openai", "langchain", "vector databases", "pinecone", "weaviate", "faiss"]
    important_skills = extract_skills_from_jd(job_description, all_skills)
    skill_score, matched_skills = skill_match_score(resume_text, important_skills)

    final_score = calculate_final_score(kw_score, sem_score, skill_score)

    return {
        "final_score": round(final_score, 2),
        "matched_skills": matched_skills,
        "important_skills": important_skills,
        "matched_keywords": matched_keywords
    }


