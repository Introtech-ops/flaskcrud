from dotenv import load_dotenv
from flask import Flask, request, jsonify

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains import load_qa_chain

# from langchain.llms import openai
from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import pdfplumber
import PyPDF2
import os
import logging
import getpass
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

load_dotenv() 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")

_cache = {
    "text_chunks": None,
    "vectorstore": None
}

greetings_dict = {
    'hello': 'Hello! How can I assist you today with the building code or any inquiries?',
    'hi': 'Hi there! What can I help you with?',
    'greetings': 'Greetings! How may I be of service?',
    'hey': 'Hey! What would you like to know about the building code?',
    'good morning': 'Good morning! How can I help you today?',
    'good evening': 'Good evening! What information are you looking for?',
    'niaje': 'Niaje! How can I assist you?',
    'vipi': 'Vipi! What do you need help with?',
    'sasa': 'Sasa! Do you have any questions for me?',
    'rieng': 'Rieng! Let me know how I can help you.'
}

def search_pdf(search_text):
    try:
        # Open the PDF file in read-binary mode
        with open('ncabc140.pdf', 'rb') as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            results = []

            # Iterate through each page in the PDF
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()  # Extract text from the page
                if search_text.lower() in text.lower():
                    results.append((page_number, text.strip()))

            if results:
                print(f"The term '{search_text}' was found in the following pages:\n")
                for page_number, context in results:
                    print(f"Page {page_number}:\n")
                    print(context[:500] + '...\n')  # Print a snippet of the context
            else:
                print(f"The term '{search_text}' was not found in the document.")
    except FileNotFoundError:
        print("The specified file was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
def extract_context_from_pdf(search_text):
    """
    Searches for a specific text in a PDF and returns matching contexts.
    """
    try:
        with open('ncabcall.pdf', 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            results = []

            # Iterate through pages to find the search text
            for page_number, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                if search_text.lower() in text.lower():
                    results.append({'page': page_number, 'content': text.strip()[:500] + '...'})

            return results
    except FileNotFoundError:
        raise FileNotFoundError("The specified file was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while processing the PDF: {e}")


## Function to check if the message contains a greeting and return a unique response
def is_greeting(message):
    message_lower = message.lower()
    for greet, response in greetings_dict.items():
        if greet in message_lower:
            return response
    return None

# Function to check if the message makes sense (simple language validation)
def validate_language(message):
    if not re.match(r"^[a-zA-Z0-9\s.,!?']+$", message):
        return False
    return True

# Function to reformat or prompt user if the message does not make sense
def reformat_message(message):
    if not validate_language(message):
        return "Your message seems unclear. Could you please rephrase it?"
    return message

# Function to search for relevant sections in the building code
def search_building_code(query, document_text):
    results = [line for line in document_text.split('\n') if query.lower() in line.lower()]
    return results if results else ["No relevant section found."]

# Load and process the PDF content
def load_nbc_content(pdf_path="ncabc140.pdf"):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            nbc_content = ""
            for page in pdf.pages:
                nbc_content += page.extract_text() + "\n"
        logging.info("PDF content loaded successfully.")
        return nbc_content
    except Exception as e:
        logging.error(f"Failed to load PDF: {e}")
        raise

def create_vector_store(text_chunks):    
    logging.info("Creating new vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    _cache['vectorstore'] = vectorstore
    return vectorstore

def initialize_text_splitter():
    logging.info("Initializing text splitter...")
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

def initialize_qa_chain():
    logging.info("Initializing QA chain...")
    prompt_template = """You are a helpful AI assistant specialized in the National Building Code 2024.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and relevant to the building code.
    Always cite the specific section numbers when referencing the code.

    Context: {context}

    Question: {question}

    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )


# Initialize everything
logging.info("Initializing the Building Code LLM system...")
def reinitialize_system():
    try:
        logging.info("Reinitializing vector store...")
        nbc_content = load_nbc_content()
        text_splitter = initialize_text_splitter()
        chunks = text_splitter.split_text(nbc_content)
        _cache['text_chunks'] = chunks
        vectorstore = create_vector_store(chunks)
        logging.info("System reinitialized successfully.")
        return vectorstore
    except Exception as e:
        logging.error(f"Reinitialization failed: {e}")
        return None

@app.route('/')
def health_chck():
    return jsonify({'status': 'healthy', 'message': 'Building Code LLM is running'})

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        # Ensure vectorstore is initialized or re-initialized
        if _cache['vectorstore'] is None:
            logging.warning("Vectorstore is not initialized, attempting to reinitialize...")
            vectorstore = reinitialize_system()
            if vectorstore is None:
                return jsonify({'error': 'Failed to reinitialize vectorstore. Please try again later.'}), 500
        else:
            vectorstore = _cache['vectorstore']

        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Check if message is a greeting and return a response if true
        greeting_response = is_greeting(user_message)
        if greeting_response:
            return jsonify({
                'response': greeting_response,
                'status': 'success'
            })
            
        user_message = reformat_message(user_message)

        logging.info(f"Received message: {user_message}")
        
        docs = vectorstore.similarity_search(user_message)
        logging.info("Similarity search completed.")

        # response = "there should be a response"
        # Generate a response (you can implement QA chain here if needed)
        response = "\n".join(docs) if docs else "Could not find relevant sections in the code."


        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        logging.error(f"Error in /api/chat: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Building Code LLM is running'})


@app.route('/api/getcontext', methods=['POST'])
def get_context():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.json
        search_text = data.get('search_text', '')
        
        if not search_text:
            return jsonify({'error': 'Both pdf_path and search_text are required.'}), 400
        
        logging.info(f"Searching for '{search_text}' in PDF: ncabcall.pdf")
        results = extract_context_from_pdf(search_text)

        if not results:
            return jsonify({'response': 'No matching text found in the document.', 'status': 'success'})
        
        logging.info(f"{results}")
        
        return jsonify({
            'response': results,
            'status': 'success'
        })

    except FileNotFoundError as fnf_error:
        logging.error(f"File error: {fnf_error}")
        return jsonify({'error': str(fnf_error), 'status': 'error'}), 404
    except Exception as e:
        logging.error(f"Error in /api/getcontext: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Error handling
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
    
  