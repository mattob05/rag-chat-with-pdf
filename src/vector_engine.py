from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

MODEL_NAME = 'BAAI/bge-m3'
DB_DIR = 'vector_db'


def load_embedding_model():
    try:
        embedder = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        return embedder
    except Exception as e:
        raise RuntimeError(e)


class VectorEngine:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_dir = os.path.join(base_dir, '..', DB_DIR)
        self.embedder = load_embedding_model()
        self.vector_store = Chroma(
            collection_name='pdf_db',
            embedding_function=self.embedder,
            persist_directory=self.db_dir)

    def add_documents(self, docs):
        self.vector_store.add_documents(docs)

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={'k':10})
    
    def search(self, question):
        retriever = self.get_retriever()
        return retriever.invoke(question)
    
    def get_all_files(self):
        data = self.vector_store.get(include=['metadatas'])
        if not data['metadatas']:
            return []
        sources = set(m.get('source', 'Unknown') for m in data['metadatas'])
        return sorted([os.path.basename(s) for s in sources])

    def delete_file_by_name(self, filename):
        data = self.vector_store.get(include=['metadatas'])
        ids_to_delete = [
            data['ids'][i] for i, m in enumerate(data['metadatas']) 
            if os.path.basename(m.get('source', '')) == filename
        ]
        
        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            return True
        return False
            