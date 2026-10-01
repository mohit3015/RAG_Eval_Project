from src.reranker import RerankingRetriever

retriever = RerankingRetriever()
results = retriever.invoke("What is online eval and how is it different from offline eval?")

print(f"Got {len(results)} results")
for r in results:
    print(r.page_content[:150])
    
    
# python -m debug_retrieval   >>>run this file standalone to test the reranker and see what it retrieves