from flask import Flask, render_template,request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/',methods=['POST'])
def predict():
    if request.method=='POST':
        data = request.form['user_input']
        print(data)
        return data
    return ""
if  __name__=="__main__":
    app.run(debug=True)