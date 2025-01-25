import logging
from flask import Flask, request, jsonify, stream_with_context, Response
from flask_cors import CORS
from llama_index.readers.file import PDFReader
from ragatouille import RAGPretrainedModel
import os
from openai import OpenAI
from settings import OPENAI_KEY, OPENAI_ORG, OPENAI_PROJECT, MODEL_NAME
from werkzeug.exceptions import HTTPException
import json

# Initialize logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

CORS(app)  # Enable CORS for all routes


data_store = {}

# Set up OpenAI client
logger.info("Setting up OpenAI client...")
try:
    client = OpenAI(api_key=OPENAI_KEY, project=OPENAI_PROJECT,
                    organization=OPENAI_ORG)
    logger.info("OpenAI client initialized successfully.")
except Exception as e:
    logger.exception(
        "Failed to initialize OpenAI client. Ensure the credentials are correct.")
    client = None


@app.errorhandler(Exception)
def handle_exception(e):
    """Global error handler."""
    logger.exception("An unexpected error occurred.")
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code
    return jsonify({"error": "An unexpected error occurred on the server."}), 500


@app.route('/', methods=['GET'])
def health():
    return "Welcome to lean alive dev API :)"


@app.route('/upload_file', methods=['POST'])
def upload_file():
    """Endpoint to upload a PDF file and index its content."""
    try:
        if 'file' not in request.files:
            logger.error("No file provided in the request.")
            return jsonify({"error": "No file provided."}), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename received.")
            return jsonify({"error": "Empty filename."}), 400

        # Save file temporarily
        temp_filepath = os.path.join("/tmp", file.filename)
        logger.debug(f"Saving file to temporary path: {temp_filepath}")
        file.save(temp_filepath)

        return jsonify({"file_path": temp_filepath, "file_name": file.filename})
    except Exception as e:
        logger.exception("An error occurred in the upload_file endpoint.")
        return jsonify({"error": str(e)}), 500


@app.route("/index", methods=['GET'])
def index_file_sse():
    @stream_with_context
    def stream_index():
        try:
            temp_filepath = request.args.get('file_path')
            file_name = request.args.get('file_name')

            if not temp_filepath or not file_name:
                raise ValueError("'file_path' and 'file_name' are required.")

            index_name = os.path.splitext(file_name)[0]

            if index_name not in data_store:

                loader = PDFReader()
                yield f"""data: {json.dumps({ "type": "text", "content": "Loading data from the PDF file..." })}\n\n"""
                documents = loader.load_data(temp_filepath)

                list_pdf_documents = [document.text for document in documents]

                data_store[index_name] = RAGPretrainedModel.from_pretrained(
                    MODEL_NAME)

                yield f"""data: {json.dumps({ "type": "text", "content": "Indexing documents..." })}\n\n"""

                data_store[index_name].index(
                    collection=list_pdf_documents,
                    index_name=index_name,
                    max_document_length=256,
                    split_documents=True,
                )

            yield f"""data: {json.dumps({ 
                "type": "data", 
                "content": {
                    "message": f"File indexed successfully! You can now ask questions...", 
                    "index_name": index_name 
                }
            })}\n\n"""
        except Exception as e:
            logger.error(e)
            yield f"""data: {json.dumps({ "type": "error", "content": {"error": "something went wrong!"} })}\n\n"""
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                yield f"""data: {json.dumps({ "type": "done", "content": "Done!" })}\n\n"""
                os.remove(temp_filepath)

    return Response(stream_index(), content_type='text/event-stream')


@app.route('/search', methods=['POST'])
def search():
    """Endpoint to perform a search over an indexed PDF document."""
    try:
        data = request.get_json()
        query = data.get('query')
        file_name = data.get('file_name')
        top_k = data.get('top_k', 5)
        system_prompt = data.get('system_prompt', """
            You are an expert in answering the question of the user.

            You will be provided with some relevant text chunks from the document the user's query is being asked from.

            Generate answer only from the provided chunks.

            All the provided chunks may not be helpful, so analyze carefully and process the respective chunk only if it is related to the user's query.

            Try to generate point-wise precise answers.

            Return with the most meaningful response in markdown format.
        """)

        index_name = os.path.splitext(file_name)[0]

        if not query or not index_name:
            logger.error("Both 'query' and 'index_name' are required.")
            return jsonify({"error": "Both 'query' and 'index_name' are required."}), 400

        logger.info(
            f"Retrieving results for query: {query} from index: {index_name}")

        if index_name not in data_store:
            return jsonify({
                "error_message": "server got refreshed, please upload the pdf again",
                "error_tracking_code": "444"
            }), 400

        results = data_store[index_name].search(
            query=query, k=top_k, index_name=index_name)

        chunks_prompt = "\nChunks -"
        for res in results:
            chunks_prompt += f"\n - {res['content']}"

        user_query = f"""
            {chunks_prompt}

            **User's query:**
            {query}
        """

        conversation = [{"role": "system", "content": system_prompt}, {
            "role": "user", "content": user_query}]

        if client:
            logger.debug(
                "Sending request to OpenAI API for chat completion...")
            completion = client.chat.completions.create(
                model='gpt-4o',
                messages=conversation
            )
            response_content = completion.choices[0].message.content
        else:
            logger.error("OpenAI client is not initialized.")
            return jsonify({"error": "OpenAI client is not initialized."}), 500

        return jsonify({
            "retrieval_result": results,
            "llm_result": response_content,
            "user_query": user_query
        }), 200
    except Exception as e:
        logger.exception("An error occurred during the search process.")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    try:
        logger.info("Starting Flask application...")
        app.run(host='0.0.0.0', debug=True, port=5464)
    except Exception as e:
        logger.exception("Failed to start Flask application.")
