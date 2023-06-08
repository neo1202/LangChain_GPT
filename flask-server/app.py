import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import OPEN_API_KEY, PINECONE_KEY, SERP_API_KEY
from demoo import initialize, process_and_store_documents, get_my_agent
from werkzeug.utils import secure_filename
import os
import tempfile

app = Flask(__name__)
CORS(app)
# Access the config variables
app.config['OPEN_API_KEY'] = OPEN_API_KEY
app.config['SERP_API_KEY'] = SERP_API_KEY
app.config['PINECONE_KEY'] = PINECONE_KEY

#https://code.visualstudio.com/docs/python/tutorial-flask#_use-a-template-to-render-a-page

llm_chat, embeddings = initialize()
my_agent = get_my_agent()

@app.route('/data')
def get_time():
    return {
        'Name':"geek",
        "Age":"22",
        "Date":"222",
        "programming":"python"
    }

@app.route('/api/endpoint', methods=['POST'])
def process_input():
    data = request.get_json()
    print(f'receive data from User input: {data}')
    user_input = data.get('inputText') #等同於 data['inputText']
    Ai_response =  my_agent.run(user_input)
    
    return jsonify({'response': Ai_response})

#python -m pip install flask 

#python -m venv .venv 
#. .venv/bin/activate  
#python -m flask run
if __name__ == '__main__': 
    app.run(debug=True, port=5000)

#ffff