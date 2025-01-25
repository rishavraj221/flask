from flask import Flask, request, jsonify
from llama_index.readers.file import PDFReader
import pymupdf
# from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient
from colbert.infra import ColBERTConfig, Run, RunConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.modeling.colbert import ColBERT
import os
import numpy as np
from utils.helpers import filter_string
# from llama_index.core import VectorStoreIndex, Document
# from llama_index.vector_stores.milvus import MilvusVectorStore

app = Flask(__name__)


# Initialize the ColBERT model
colbert_config = ColBERTConfig()
checkpoint = Checkpoint("colbert-ir/colbertv2.0",
                        colbert_config=colbert_config)
colbert = checkpoint.model

client = MilvusClient('./milvus_demo.db')


fields = [
    FieldSchema(name="id", dtype=DataType.INT64,
                is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)
]
db_schema = CollectionSchema(
    fields=fields, description="Collection for storing text embeddings")


def preprocess_pdf(file_path):
    reader = PDFReader()
    documents = reader.load_data(file_path)  # Read the entire PDF into chunks
    text_chunks = [document.text for document in documents]
    return text_chunks

# Function to store embeddings in Milvus


def generate_embeddings(chunks):
    data_to_insert = []
    # embedding_matrix = checkpoint.docFromText(chunks)
    # print(embedding_matrix.size())
    for i in range(len(chunks)):
        chunk = chunks[i]
        embedding_matrix = checkpoint.docFromText(
            [chunk])
        embedding = embedding_matrix[0].tolist()

        document = {
            "text": chunk,
            "embedding": embedding[0]
        }

        print(len(embedding), len(embedding[0]))

        data_to_insert.append(document)

    return data_to_insert

# Route to upload and process PDF


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():

    file = request.files['file']
    file_path = os.path.join("/tmp", file.filename)
    file.save(file_path)

    chunks = preprocess_pdf(file_path)

    for i in range(len(chunks)):
        print(f"chunk {i+1}")

        chunk = chunks[i]

        print(
            f"length of chunk {i+1} is {len(chunk)}, type of chunk is {type(chunk)}")

    data_to_store = generate_embeddings(chunks)

    collection_name = filter_string(file.filename)

    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name, dimension=128, schema=db_schema)

    res = client.insert(collection_name=collection_name, data=data_to_store)

    import pprint
    pprint.pp(res)

    return jsonify({"message": "Done!"})


# @app.route('/search', methods=['GET'])
# def search():
#     q = request.args.get('q')

#     query_embedding = checkpoint.docFromText([q])[0].tolist()

#     search_result = collection.search(
#         data=[query_embedding],
#         anns_field="embedding",
#         param={"metric_type": "IP", "params": {"nprobe": 10}},
#         limit=10
#     )

#     results = [
#         {
#             "text": hit.entity.get("text"),
#             "distance": hit.distance
#         }
#         for hit in search_result[0]
#     ]

#     return jsonify({"results": results})


if __name__ == '__main__':

    app.run(debug=False, port='5544')
