from flask import Flask
from flask import render_template,jsonify,request
from faqengine import *
import random


app = Flask(__name__)
app.secret_key = '12345'

faqslist = ["data/dataset.csv"]
faqmodel = FaqEngine(faqslist)

def get_response(user_message):
    return faqmodel.query(user_message)

@app.route('/')
def hello_world():
    return render_template('home.html')

@app.route('/chat',methods=["POST"])
def chat():
    try:
        user_message = request.form["text"]
        response_text = get_response(user_message)
        return jsonify({"status":"success","response":response_text})
    except Exception as e:
        print(e)
        return jsonify({"status":"success","response":"माफ करा मी तसे करण्यास प्रशिक्षित नाही"})

if __name__ == "__main__":
    app.run(port=8080)
