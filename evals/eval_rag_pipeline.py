# eval_rag_pipeline.py

import os
from dotenv import load_dotenv
os.environ["DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE"] = "180"   # 3 minute per task

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
# from deepeval.models import OpenRouterModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import RagPipeline
from src.custom_llm import OpenRouterModel 
from evals.harness import load_goldens, summarize_by_metric, print_summary

load_dotenv()

GOLDEN_PATH = "goldens/faithfulness_dataset.json"   # reuse the queries
JUDGE_MODEL = OpenRouterModel(
    model_name="nvidia/nemotron-3-super-120b-a12b:free",   # model → model_name
    api_key=os.getenv("RAG_API_KEY"),
)
THRESHOLD = 0.7


def run(rag):
    # 1. LOAD queries (we only need the queries --- context comes from the pipeline now)
    goldens = load_goldens(GOLDEN_PATH)[:2]   # TEMP: sirf 2 goldens, quota bachane ke liye



    # 2. RUN THE INJECTED PIPELINE per query, build a test case from LIVE output
    test_cases = []
    for g in goldens:
        result = rag.invoke(g["query"])          # retrieve -> rerank -> generate

        test_cases.append(
            LLMTestCase(
                input=g["query"],
                actual_output=result["answer"],       # what the generator produced
                retrieval_context=result["context"],  # what the RETRIEVER returned
            )
        )


    # 3. THE THREE TRIAD METRICS
    metrics = [
        ContextualRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        FaithfulnessMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=JUDGE_MODEL, include_reason=True),
    ]


    # 4. EVALUATE
    result = evaluate(test_cases=test_cases, metrics=metrics)
    return summarize_by_metric(result)


def run_local():
    """Standalone convenience: build the pipeline, then run."""
    return run(RagPipeline())


if __name__ == "__main__":
    print_summary("rag_pipeline", run_local())
    
    
    # python -m evals.eval_rag_pipeline  >>>run the RAG pipeline evaluation