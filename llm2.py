from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import faiss
import pdfplumber
import os
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load Llama model and tokenizer
logging.info("Loading Llama model and tokenizer...")
llama_model_name = "meta-llama/Llama-2-7b-hf"  # Adjust this to the Llama model you're using
tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name)

# Initialize sentence transformer for embeddings
logging.info("Loading SentenceTransformer for embeddings...")

# hf_token = "your_huggingface_api_token"
hf_token = "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"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(llama_model_name, use_auth_token=hf_token)

# Cache for vector store and text chunks
_cache = {
    "text_chunks": None,
    "vectorstore": None
}

# Greetings dictionary
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

def is_greeting(message):
    message_lower = message.lower()
    for greet, response in greetings_dict.items():
        if greet in message_lower:
            return response
    return None

def validate_language(message):
    if not re.match(r"^[a-zA-Z0-9\s.,!?']+$", message):
        return False
    return True

def reformat_message(message):
    if not validate_language(message):
        return "Your message seems unclear. Could you please rephrase it?"
    return message

def search_building_code(query, document_text):
    results = [line for line in document_text.split('\n') if query.lower() in line.lower()]
    return results if results else ["No relevant section found."]

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
    embeddings = embedding_model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    _cache['vectorstore'] = {
        "index": index,
        "chunks": text_chunks
    }
    return _cache['vectorstore']

def initialize_text_splitter():
    logging.info("Splitting text into chunks...")
    chunk_size = 1000
    chunk_overlap = 200
    chunks = []
    text = _cache['text_chunks']
    for i in range(0, len(text), chunk_size - chunk_overlap):
        chunks.append(text[i:i+chunk_size])
    return chunks

def reinitialize_system():
    try:
        logging.info("Reinitializing vector store...")
        nbc_content = load_nbc_content()
        _cache['text_chunks'] = nbc_content
        chunks = initialize_text_splitter()
        vectorstore = create_vector_store(chunks)
        logging.info("System reinitialized successfully.")
        return vectorstore
    except Exception as e:
        logging.error(f"Reinitialization failed: {e}")
        return None

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
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
        
        greeting_response = is_greeting(user_message)
        if greeting_response:
            return jsonify({'response': greeting_response, 'status': 'success'})

        user_message = reformat_message(user_message)
        logging.info(f"Received message: {user_message}")
        
        query_vector = embedding_model.encode([user_message])
        distances, indices = vectorstore["index"].search(query_vector, k=3)
        docs = [vectorstore["chunks"][i] for i in indices[0]] if indices.size > 0 else ["No relevant sections found."]
        
        response = "\n".join(docs)

        return jsonify({'response': response, 'status': 'success'})
    except Exception as e:
        logging.error(f"Error in /api/chat: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
