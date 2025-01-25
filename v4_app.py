from flask import Flask, request, jsonify
import pymupdf
import os
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from utils.helpers import top_k_matches


app = Flask(__name__)

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
CHUNK_SIZE = 64
OVERLAP = 8

ckpt = Checkpoint(name=MODEL_NAME, colbert_config=ColBERTConfig())

data_store = {}


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():

    global data_store

    file = request.files.get('file')
    file_path = os.path.join('/tmp', file.filename)
    file.save(file_path)

    pdf = pymupdf.open(file_path)
    text = ''
    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)
        text += page.get_text()

    word_chunks = []
    words = text.split()

    chunk_size = int(request.args.get('chunk_size', CHUNK_SIZE))
    overlap = int(request.args.get('overlap', OVERLAP))

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        word_chunks.append(chunk)

    if os.path.exists(file_path):
        os.remove(file_path)

    embeddings = ckpt.queryFromText(word_chunks, bsize=16)

    data_store[file.filename] = {
        "docs_length": len(word_chunks),
        "embeddings_size": str(embeddings.size()),
        "docs": word_chunks,
        "embeddings": embeddings,
    }

    return "Done!"


@app.route('/data', methods=["GET"])
def get_global_data():
    data_to_return = {}

    for file_name in data_store:
        data = data_store[file_name]
        data_to_return[file_name] = {
            "docs_length": data.get('docs_length'),
            "embeddings_size": data.get('embeddings_size'),
        }

    return jsonify(data_to_return)


@app.route('/search', methods=["GET"])
def search():

    global data_store

    q = request.args.get('q')
    top_k = request.args.get('top_k', 10)
    file_name = request.args.get('file_name')

    doc_data = data_store[file_name] if file_name in data_store else {}

    if not doc_data:
        return jsonify({"error": "data not there!, please index it again"}), 400

    query_embedding = ckpt.docFromText(
        [q], bsize=16)[0][0].detach().cpu().numpy()
    embeddings = doc_data.get('embeddings').detach().cpu().numpy()

    matches = top_k_matches(query_embedding,
                            embeddings, int(top_k))

    results = []
    chunk_set = set()
    for score, chunk_index in matches:

        if chunk_index in chunk_set:
            continue

        results.append({
            "text": doc_data.get('docs')[chunk_index],
            "score": score.tolist()
        })

        chunk_set.add(chunk_index)

    return jsonify({
        "results": results
    })


if __name__ == "__main__":
    app.run(debug=False, port='5544')
