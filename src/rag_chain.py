from .prompts import get_rag_prompt


class RAGChain:
    def __init__(self, vector_engine, reranker, llm):
        self.vector_engine = vector_engine
        self.reranker = reranker
        self.llm = llm

    def get_response(self, question):
        top_frags = self.vector_engine.search(question)
        final_frags = self.reranker.rerank(question, top_frags)
        context = '\n\n'.join([doc.page_content for doc in final_frags])
        prompt = get_rag_prompt()
        formatted_prompt = prompt.format(context=context, question=question)
        res = self.llm.get_llm().invoke(formatted_prompt)
        return res
