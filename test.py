import requests

url = "https://api.hyperbolic.xyz/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJPUTZRR0pVYVp0TUZJcVNURTZ3R2tqOG54YUsyIiwiaWF0IjoxNzQ5MjQwNzU4fQ.IzQ1XJ7NfEsPabNsPAPXX-GKrGT2iEPyS7pfk4nMDv0"
}
data = {
    "messages": [{
      "role": "user",
      "content": "What can I do in SF?"
    }],
    "model": "deepseek-ai/DeepSeek-R1-0528",
    "max_tokens": 508,
    "temperature": 0.1,
    "top_p": 0.9
}

response = requests.post(url, headers=headers, json=data)
print(response.json())