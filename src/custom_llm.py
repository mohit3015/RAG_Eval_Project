import time
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI, RateLimitError

class OpenRouterModel(DeepEvalBaseLLM):
    def __init__(self, model_name="nvidia/nemotron-3-super-120b-a12b:free", api_key=None):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        max_retries = 3                      # 5 ki jagah — kam retries, jaldi fail ho jayega agar genuinely stuck hai
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content
            except RateLimitError:
                if attempt == max_retries - 1:
                    raise
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s...
                print(f"Rate limited, retrying in {wait}s...")
                time.sleep(wait)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name
    
    
    # def get_model_type(self):
    #     return "OpenRouterModel"
    # python -m evals.custom_llm   >>>run this file standalone to test the OpenRouterModel class