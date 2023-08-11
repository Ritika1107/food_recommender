# Import required libraries
from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


# Create a Flask application
app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html') # Return the HTML file that's in the templates folder






# Start the Flask application, allowing for debugging
if __name__ == '__main__':
    app.run(debug=True)


# Function to generate recipes based on user input
def generate_recipes(user_input):
    # Encode the user input as integers for the model
    input_ids = tokenizer.encode(user_input, return_tensors='pt')
    # Generate text from the model (e.g., recipes) with specific constraints
    output = model.generate(input_ids, max_length=150, num_return_sequences=5, no_repeat_ngram_size=2)
    # Decode the generated output back into human-readable text
    recipes = [tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in output]
    return recipes

# Inside your '/generate' route in app.py
@app.route('/generate', methods=['POST'])
def generate():
    # Retrieve the user input fields from the JSON request body
    ingredients = request.json.get('ingredients')
    time = request.json.get('time')
    diet = request.json.get('diet')
    allergies = request.json.get('allergies')

    # Check for required fields
    if not all([ingredients, time, diet]):
        return jsonify({'error': 'Ingredients, time, and diet are required'}), 400

    # Construct a user_input string or any processing logic that fits your needs
    user_input = f"Ingredients: {ingredients}, Time: {time}, Diet: {diet}, Allergies: {allergies}"

    # Call the function to generate recipes based on the user input
    recommendations = generate_recipes(user_input)
    # Return the generated recipes as JSON
    return jsonify(recommendations)

