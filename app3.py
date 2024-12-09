import streamlit as st
from PyPDF2 import PdfReader  # Read the PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Split text into smaller chunks
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Generate embeddings using Google AI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS  # Facebook AI Similarity Search
from langchain_google_genai import ChatGoogleGenerativeAI  # Chat capabilities using Google Gen AI
from langchain.chains.question_answering import load_qa_chain  # Load question-answering chain
from langchain.prompts import PromptTemplate  # Define the prompt template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the prompt template
prompt_template = """
Provide an answer only if it directly relates to Wilya's context, products, or services. Wilya offers a supervisor web app for managing shifts and a worker mobile app for shift acceptance and other tasks. Additional information about Wilya can be found at www.wilya.com.

If the question is not relevant to Wilya or its context, respond with:
"I am here to assist with queries specifically related to Wilya and its offerings. This question does not pertain to Wilya or the provided company documents. Please feel free to ask questions directly relevant to Wilya."

Context:
{context}

Question:
{question}

Answer:
"""

# Function to extract text from all PDFs in a folder
def get_pdf_text(pdf_folder):
    text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Processing file: {filename}")  # Debug statement
            filepath = os.path.join(pdf_folder, filename)
            pdf_reader = PdfReader(filepath)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    print(f"Total extracted text length: {len(text)}")  # Debug statement
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)
    print(f"Total chunks created: {len(text_chunks)}")  # Debug statement
    return text_chunks

# Function to create and save FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    print("Vector store created with chunks:", len(text_chunks))  # Debug statement
    vector_store.save_local("faiss_index")
    print("FAISS index saved locally.")  # Debug statement

# Function to load the conversational chain
def get_conversational_chain():
    prompt_temp = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model='gemini-pro')
    print("Conversational chain loaded.")  # Debug statement
    return load_qa_chain(model, prompt=prompt_temp)

# Function to process user input and generate a response
def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(question)
    print(f"Number of relevant documents found: {len(docs)}")  # Debug statement

    if not docs:
        st.write("I am here to assist with queries specifically related to Wilya and its offerings. This question does not pertain to Wilya or the provided company documents. Please feel free to ask questions directly relevant to Wilya.")
        return

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    print("Response generated.")  # Debug statement
    st.write(response['output_text'])

# Main application logic
pdf_folder = "pdf"  # Folder where PDFs are stored
if not os.path.exists("faiss_index"):  # Only process if the index doesn't already exist
    print("FAISS index not found. Processing PDFs...")  # Debug statement
    raw_text = get_pdf_text(pdf_folder)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
else:
    print("FAISS index already exists. Skipping PDF processing.")  # Debug statement

# Streamlit UI
st.set_page_config(page_icon="📜", page_title="Company Chatbot")
st.header("Ask Questions About the Company")
st.title("Ask Your Questions")

user_question = st.text_input("Enter a company-related question:")
if user_question:
    print(f"User question received: {user_question}")  # Debug statement
    user_input(user_question)
