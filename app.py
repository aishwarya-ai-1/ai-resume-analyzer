import streamlit as st
import PyPDF2
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nlp = spacy.load("en_core_web_sm")

st.title("AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

job_description = st.text_area("Enter Job Description")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if uploaded_file is not None and job_description != "":
    resume_text = extract_text_from_pdf(uploaded_file)

    text = [resume_text, job_description]

    cv = CountVectorizer()
    matrix = cv.fit_transform(text)

    similarity = cosine_similarity(matrix)[0][1]

    score = round(similarity * 100, 2)

    st.subheader("Resume Match Score")
    st.write(score,"%")

    doc = nlp(resume_text)

    skills = []

    for token in doc:
        if token.pos_ == "NOUN":
            skills.append(token.text)

    st.subheader("Extracted Keywords")
    st.write(set(skills))