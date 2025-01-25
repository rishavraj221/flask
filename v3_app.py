from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from flask import Flask, request, jsonify
from flask_cors import CORS
import pymupdf
import os
from utils.helpers import filter_alphanumeric
# from pinecone import Pinecone, ServerlessSpec
import time
# from rerankers import Reranker

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

CORS(app)  # Enable CORS for all routes


MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
PINECONE_API_KEY = "pcsk_4FcXEt_K1AGWuZuYGT482gDKngBZU4HpftespNpGF9RpWUzpMg2to6jTggadcSL8sgwpqX"
PINECONE_HOSTED_REGION = "us-east-1"

# pc = Pinecone(
#     api_key=PINECONE_API_KEY)

# ckpt = Checkpoint("answerdotai/answerai-colbert-small-v1",
#                   colbert_config=ColBERTConfig())

# ranker = Reranker(model_name=MODEL_NAME, model_type="colbert")


@app.route('/', methods=['GET'])
def health():
    return "Welcome to lean alive test API :)"


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():

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
    chunk_size = 64
    overlap = 8
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        word_chunks.append(chunk)

    # temp
    # q = request.args.get('q')
    # res = ranker.rank(query=q, docs=word_chunks)

    config = ColBERTConfig(
        doc_maxlen=512,
        nbits=2
    )

    indexer = Indexer(
        checkpoint="answerdotai/answerai-colbert-small-v1",
        config=config
    )

    index_name = filter_alphanumeric(file.filename)

    res = indexer.index(
        name=index_name, collection=word_chunks, overwrite=True)

    # embeddings = []
    # for chunk in word_chunks:

    #     embedded_query = ckpt.queryFromText([chunk], bsize=16)

    #     print(embedded_query[0].size())  # getting torch.Size([32, 96])

    #     embeddings.append(embedded_query[0].tolist())

    # # add db code here......
    # pc.create_index(
    #     name=index_name,
    #     dimension=3072,
    #     metric="cosine",
    #     spec=ServerlessSpec(
    #         cloud="aws",
    #         region=PINECONE_HOSTED_REGION
    #     )
    # )

    # pc_index = pc.Index(index_name)

    # for i, unflatten_embedding in enumerate(embeddings):
    #     id = f"{index_name}-{i}"
    #     metadata = {"text": word_chunks[i]}
    #     embedding = [
    #         item for sublist in unflatten_embedding for item in sublist]
    #     embedding = [float(item) for item in embedding]

    #     pc_index.upsert(
    #         vectors=[(id, embedding, metadata)], namespace=file.filename)

    # print(f"describe index : {pc_index.describe_index_stats()}")

    return jsonify({
        "file_name": file.filename,
        "len_word_chunks": len(word_chunks),
        # "results": res.dict()
    })


@app.route('/search', methods=["GET"])
def search():
    q = request.args.get('q')
    file_name = request.args.get('file_name')

    # index_name = filter_alphanumeric(file_name)

    # config = ColBERTConfig(query_maxlen=32)
    # searcher = Searcher(index=index_name, config=config)

    # results = searcher.search(q, k=10)

    # embedded_query = ckpt.queryFromText([q], bsize=16)

    # print(embedded_query[0].size())  # getting torch.Size([32, 96])

    # unflatten_embedding = embedded_query[0].tolist()

    # embedding = [item for sublist in unflatten_embedding for item in sublist]
    # embedding = [float(item) for item in embedding]

    # pc_index = pc.Index(index_name)

    # results = pc_index.query(
    #     namespace=file_name,
    #     vector=embedding,
    #     top_k=10,
    #     include_metadata=True
    # )

    return jsonify({})


if __name__ == "__main__":
    app.run(debug=False, port="5544")
