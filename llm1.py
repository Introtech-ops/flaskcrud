from dotenv import load_dotenv
from flask import Flask, request, jsonify

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.chains import load_qa_chain

# from langchain.llms import openai
from langchain.prompts import PromptTemplate

import pdfplumber
import os
import logging
import getpass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

load_dotenv() 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")



from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

model = ChatOpenAI(model="gpt-4o-mini")


# Cache for text chunks and vector store
_cache = {
    "text_chunks": None,
    "vectorstore": None
}

# Load and process the PDF content
def load_nbc_content(pdf_path="National_Building_Code_2024.pdf"):
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

# Create embeddings and vector store (with caching)
def create_vector_store(text_chunks):
    if _cache['vectorstore']:
        logging.info("Using cached vector store.")
        return _cache['vectorstore']
    
    logging.info("Creating new vector store...")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    _cache['vectorstore'] = vectorstore
    return vectorstore

# Initialize text splitter
def initialize_text_splitter():
    logging.info("Initializing text splitter...")
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

# Initialize QA chain
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

try:
    nbc_content = load_nbc_content()
    text_splitter = initialize_text_splitter()
    chunks = text_splitter.split_text(nbc_content)
    _cache['text_chunks'] = chunks
    vectorstore = create_vector_store(chunks)
    qa_chain = initialize_qa_chain()
except Exception as e:
    logging.error(f"Initialization failed: {e}")
    
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        logging.info(f"Received message: {user_message}")

        # Search for relevant documents
        docs = vectorstore.similarity_search(user_message)
        logging.info("Similarity search completed.")

        # Generate response
        # response = qa_chain.run(input_documents=docs, question=user_message)
        response = "there should be a response"

        return jsonify({
            'response': response,
            'status': 'success'
        })

    except Exception as e:
        logging.error(f"Error in /api/chat: {e}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Building Code LLM is running'})



# Error handling
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Route not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
