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
#CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# Access the config variables
app.config['OPEN_API_KEY'] = OPEN_API_KEY
app.config['SERP_API_KEY'] = SERP_API_KEY
app.config['PINECONE_KEY'] = PINECONE_KEY

#https://code.visualstudio.com/docs/python/tutorial-flask#_use-a-template-to-render-a-page

llm_chat, embeddings = initialize()
my_agent = get_my_agent()
print(f"\n --- \n Agent prompt:\n {my_agent.agent.llm_chain.prompt}\n")
print(f"Agent Output Parser: {my_agent.agent.llm_chain.prompt.output_parser}\n---\n")

@app.route('/data')
def get_time():
    return {
        'Name':"geek",
        "Age":"22",
        "Date":"222",
        "programming":"python"
    }

@app.route('/upload_doc', methods=['POST'])
#Two possibilities -> a single folder or 1~multiple files
def upload_documents():
    responses = []
    print(f'\nfile objects that I receive in this upload:{request.files}')
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request.'}), 400

    files = request.files.getlist('file')  # Get the list of files
    if files[0].filename.endswith('/'):  # 處理folder成一個list of files 好像用不到
        print('haha')
        folder_path = files[0].filename
        file_names = os.listdir(folder_path)
        files = [open(os.path.join(folder_path, file_name), 'rb') for file_name in file_names]

    for file in files:
        print('\nthis file:', file)
        if file.filename == '':
            responses.append({'message': '''There's a bad file.''', 'code':400})
        # 保存文件到臨時位置
        temp_file = tempfile.NamedTemporaryFile(delete=False) #創建一個位置以及位置路徑
        file.save(temp_file.name)
        print("Temporary file path:", temp_file.name)
        try:
            process_and_store_documents([temp_file.name])  
            responses.append({
                'message': f'File uploaded and processed successfully to Temporary space {temp_file.name}','code':200})
        except Exception as e:
            app.logger.debug('Debug message')
            responses.append({'message': f'Error processing file: {str(e)}', 'code':500})
        finally:
            # 刪除臨時文件
            os.remove(temp_file.name)
    responses.append('end message:file upload successfully finished')
    return jsonify({'responses': responses}), 200

@app.route('/get_answer', methods=['POST'])
def process_input():
    data = request.get_json()
    print(f'receive data from User input: {data}')
    user_input = data.get('inputText') #等同於 data['inputText']
    try:
        Ai_response = my_agent.run(user_input)
    except Exception as e:
        Ai_response = str(e)
        print(f'The error message is here: {e}')
        if Ai_response.startswith("Could not parse LLM output: `"):
            Ai_response = Ai_response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    #Ai_response = my_agent.run(user_input)
    
    return jsonify({'response': Ai_response})

#. .venv/bin/activate  
#python -m flask run
if __name__ == '__main__': 
    app.run(debug=True, port=5000)