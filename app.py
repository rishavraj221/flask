from flask import Flask, request, jsonify
from llama_index import LLMIndex

app = Flask(__name__)

# Initialize LLM Index
llm_index = LLMIndex()

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = llm_index.query(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)