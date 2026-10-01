import os
from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OpenRouterModel


load_dotenv()

# --- Test case 1: a good answer (should PASS) ---
case_1 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

# --- Test case 2: an off-topic answer (should FAIL) ---
case_2 = LLMTestCase(
    input="What is the capital of France?",
    actual_output="France is a beautiful country famous for its food and wine.",
)


judge = OpenRouterModel(
    model="nvidia/nemotron-3-super-120b-a12b:free",   # model
    api_key=os.getenv("RAG_API_KEY"),
)

# --- One metric, judged by an LLM (pinned for reproducibility) ---
metric = AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True)


# --- Run BOTH cases through the metric, with a printed report ---
evaluate(test_cases=[case_1, case_2], metrics=[metric])


# python -m deepeval_intro   >>>run this file standalone to test the deepeval framework