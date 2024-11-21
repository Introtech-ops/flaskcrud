from flask import Flask, render_template, jsonify, request, url_for, redirect, flash, request

import openai
import re
import pdfplumber

from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

products = [
	{'id': 1, 'name': 'laptop', 'price': 1000, 'quantity': 12},
	{'id': 2, 'name': 'Samsung A12s', 'price': 1200, 'quantity': 8}
]

product_counter = len(products)+1

app.secret = os.getenv("APP_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")

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

# Load the National Building Code PDF and extract content
def load_building_code():
    try:
        with pdfplumber.open('National_Building_Code_2024.pdf') as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Check if page text is not None
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""

building_code_text = load_building_code()

## Function to check if the message contains a greeting and return a unique response
def is_greeting(message):
    message_lower = message.lower()
    for greet, response in greetings_dict.items():
        if greet in message_lower:
            return response
    return None

# Function to check if the message makes sense (simple language validation)
def validate_language(message):
    # Simple check to ensure message contains alphabetical characters
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


# Main chat endpoint

# trial on lang chain

@app.route('/langchain/openai')
def langChain():
    llm = ChatOpenAI()
    result = llm.invoke("Hello World!")
    
    
    return jsonify({'resopnse': result})


# end trail on lang chain
@app.route('/api/ncabd/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Step 1: Check for greetings
        greeting_response = is_greeting(user_message)
        if greeting_response:
            return jsonify({'response': greeting_response})

        # Step 2: Validate and reformat the message if needed
        user_message = reformat_message(user_message)
        if user_message == "Your message seems unclear. Could you please rephrase it?":
            return jsonify({'response': user_message})

        # Step 3: Search for relevant sections in the National Building Code
        code_results = search_building_code(user_message, building_code_text)
        highlighted_results = "\n".join(code_results)

        # Step 4: Build an LLM-friendly prompt if applicable
        llm_prompt = f"The user asked: '{user_message}'. Here are relevant highlights:\n{highlighted_results}\nSuggest appropriate actions or guidance."

        # Generate response using OpenAI
        response = openai.Completion.create(
            engine="gpt-3.5-turbo",
            prompt=llm_prompt,
            max_tokens=200,
            n=1,
            stop=None,
            temperature=0.7
        )

        bot_response = response.choices[0].text.strip()
        return jsonify({'response': bot_response})

    except Exception as e:
        return jsonify({'error Exception': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html', products=products)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/hello', methods = ['GET', 'POST'])
def home():
    if(request.method == 'GET'):
        data = "hello world"
        return jsonify({'data': data})
    
@app.route('/square/<int:num>', methods=['GET'])
def square(num):
    return jsonify({'data': num**2})

@app.route('/product/add', methods=['GET', 'POST'])
def add_product():
    global product_counter
    if(request.method == 'POST'):
        name = request.form['name']
        price = float(request.form['price'])
        quantity = int(request.form['quantity'])
        
        new_product = {'id': product_counter, 'name': name, 'price': price, 'quantity': quantity}
        products.append(new_product)
        product_counter += 1
        
        flash('Product added successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('add_product.html')
        
@app.route('/ncabcllm', methods=['GET', 'POST'])
def ncabc_llm():
    return render_template('ncabchat.html')
        
@app.route('/product/edit/<int:product_id>', methods=['GET', 'POST'])
def edit_product(product_id):
    """Edit an existing product"""
    product = next((p for p in products if p['id'] == product_id), None)
    if not product:
        flash('Product not found!', 'danger')
        return redirect(url_for('index'))

    if request.method == 'POST':
        product['name'] = request.form['name']
        product['price'] = float(request.form['price'])
        product['quantity'] = int(request.form['quantity'])

        flash('Product updated successfully!', 'success')
        return redirect(url_for('index'))

    return render_template('edit_product.html', product=product)

@app.route('/product/delete/<int:product_id>', methods=['POST'])
def delete_product(product_id):
    """Delete a product"""
    global products
    products = [p for p in products if p['id'] != product_id]

    flash('Product deleted successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)