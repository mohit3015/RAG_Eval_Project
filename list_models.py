# Available free models check

import requests

url = "https://openrouter.ai/api/v1/models"

response = requests.get(url, timeout=20)
response.raise_for_status()

models = response.json()["data"]

for model in models:
    model_id = model["id"]

    if model_id.endswith(":free"):
        print(model_id)
        
# python -m list_models   >>>run this file standalone to see the available free models





# Retriever (Contextual Recall/Precision)
# Generator isolation (Faithfulness/AnswerRelevancy)
# Full pipeline (Contextual Relevancy/Faithfulness/AnswerRelevancy)
# Application quality (Correctness/Completeness/Style)
# Safety (Toxicity, PII/Prompt/Content Leakage, Scope Adherence)