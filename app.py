import os
import numpy as np
import faiss
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API key
configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ''.join(page.extract_text() for page in reader.pages)
    return text

# Function to split text into sections
def split_into_sections(text, section_length=500):
    words = text.split()
    sections = [' '.join(words[i:i + section_length]) for i in range(0, len(words), section_length)]
    return sections

# Function to embed text using Sentence Transformers
def embed_text(text):
    return model.encode(text, convert_to_tensor=True)

# Function to create a FAISS index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

# Function to query the Gemini model
def query_gemini(question, history=[]):
    model = GenerativeModel("gemini-1.5-pro")
    chat = model.start_chat(history=history)
    response = chat.send_message(question, stream=True)
    response.resolve()  # Ensure response is fully resolved
    return response, chat.history

# Function to perform the query
def perform_query(query, index, sections):
    query_embedding = embed_text(query)
    D, I = index.search(np.array([query_embedding]), k=3)
    docs = [sections[i] for i in I[0]]
    combined_docs = " ".join(docs)
    question = f"{combined_docs}\n\n{query}"
    response, chat_history = query_gemini(question, history=[])
    response_text = "".join(chunk.text for chunk in response)
    return response_text

# Streamlit app
def main():
    st.title('Personalized Chatbot using Streamlit')

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        st.text("PDF file uploaded successfully!")

        # Process PDF file
        text = extract_text_from_pdf(uploaded_file)
        sections = split_into_sections(text)
        embeddings = embed_text(sections)
        index = create_faiss_index(embeddings)

        # User query input
        query = st.text_area("Enter your query:")
        if st.button("Ask"):
            if query:
                response_text = perform_query(query, index, sections)
                st.text("Gemini Response:")
                st.text(response_text)

if __name__ == '__main__':
    main()
