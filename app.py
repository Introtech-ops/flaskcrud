from flask import Flask, render_template, jsonify, request, url_for, redirect, flash
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)

products = [
	{'id': 1, 'name': 'laptop', 'price': 1000, 'quantity': 12},
	{'id': 2, 'name': 'Samsung A12s', 'price': 1200, 'quantity': 8}
]

product_counter = len(products)+1

app.secret_key = 'HJBVSDAJHBGJHBFSHM'

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
    

     