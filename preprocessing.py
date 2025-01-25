
from llama_index.readers.file import PDFReader
from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


class Preprocessing:
    def __init__(self, model_name="colbert-ir/colbertv2.0"):
        colbert_config = ColBERTConfig()
        self.checkpoint = Checkpoint(
            name=model_name, colbert_config=colbert_config)

    def preprocess_pdf(self, file_path):
        reader = PDFReader()
        documents = reader.load_data(file_path)
        text_chunks = [document.text for document in documents]
        return text_chunks

    def generate_embeddings(self, chunks):
        embedding_matrix = self.checkpoint.docFromText(chunks)

        return embedding_matrix
