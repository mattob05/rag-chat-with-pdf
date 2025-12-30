from langchain_community.cross_encoders import HuggingFaceCrossEncoder

MODEL_NAME = 'BAAI/bge-reranker-base'

def load_reranker_model():
    try:
        embedder = HuggingFaceCrossEncoder(
            model_name=MODEL_NAME,
            model_kwargs={'device':'cpu'},
        )

        return embedder
    except Exception as e:
        raise RuntimeError(e)
    

class Reranker:
    def __init__(self):
        self.reranker = load_reranker_model()

    def rerank(self, question, docs):
        pairs = []
        for doc in docs:
            pairs.append((question, doc.page_content))
        scores = self.reranker.score(pairs)
        final_scores = list(zip(docs, scores))
        final_sorted = sorted(final_scores, key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in final_sorted[:3]]