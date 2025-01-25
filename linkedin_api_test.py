from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

LINKEDIN_API_URL = 'https://api.linkedin.com/v2/search'
ACCESS_TOKEN = 'YOUR_ACCESS_TOKEN_HERE'

@app.route('/search_candidates', methods=['GET'])
def search_candidates():
    job_title = request.args.get('role', 'software developer')  # Default to 'software developer'
    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }
    params = {
        'keywords': job_title,
        'count': 10
    }
    response = requests.get(LINKEDIN_API_URL, headers=headers, params=params)

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({'error': 'Unable to fetch candidates'}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)