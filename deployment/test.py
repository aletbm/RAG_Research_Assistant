import requests
from rich.console import Console
from rich.markdown import Markdown

API_URL = "http://localhost:8080/rag"

sample_input = {"query": "What is Transformers?", "categories": ["cs.CL"]}

response = requests.post(API_URL, json=sample_input)

# print(f"Status code: {response.status_code}")

print(f"\nQuery: {sample_input['query']}\n")

console = Console()
md = Markdown(response.json()["response"])
console.print(md)

# uvicorn scripts.main:app --host 0.0.0.0 --port 8080 --reload
# python ./scripts/test.py
