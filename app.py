from flask import Flask, render_template, request
from predict_spam import load_tokenizer, load_trained_model, preprocess_input

app = Flask(__name__)

# Paths to your trained model and tokenizer
model_path = r'C:\Users\vvaru\Documents\email_classification\spam_classifier_model.h5'
tokenizer_path = r'C:\Users\vvaru\Documents\email_classification\tokenizer.pkl'

# Load the tokenizer and model
tokenizer = load_tokenizer(tokenizer_path)
model = load_trained_model(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        
        # Tokenize and pad the user input
        user_input_padded = preprocess_input(user_input, tokenizer, model.input_shape[1])

        # Predict whether the input is spam or not
        prediction = model.predict(user_input_padded)
        predicted_class = "spam" if prediction[0][0] >= 0.5 else "not spam"

        return render_template('index.html', result=predicted_class, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
