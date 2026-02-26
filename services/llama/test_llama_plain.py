import requests

LLAMA_URL = "http://localhost:8090/completion"

payload = {
    "prompt": "User: Say hello and confirm you are llama.cpp running locally.\nAssistant:",
    "n_predict": 80,
    "temperature": 0.2,
    "stop": ["User:"]
}

r = requests.post(LLAMA_URL, json=payload, timeout=60)
r.raise_for_status()

print("\n=== RESPONSE ===\n")
print(r.json().get("content", r.json()))