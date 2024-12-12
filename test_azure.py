import requests

# Configuração do endpoint e chave de API
endpoint_url = (
    "https://ferre-m4lr6wu8-francecentral.openai.azure.com/"
    "openai/deployments/Ministral-3B/chat/completions?api-version=2024-05-01-preview"
)
api_key = "DiObe3mMzlO2LLvv3GVjvDxRkhEaYZ9v92rK4WA8HDcfJFG9ToUtJQQJ99ALAC5T7U2XJ3w3AAAAACOGO8RQ"

# Dados de entrada para o modelo
input_data = {
    "messages": [
        {"role": "system", "content": "Você é um assistente útil."},
        {"role": "user", "content": "Olá! Quem é você?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
}

# Cabeçalhos da requisição
headers = {
    "api-key": api_key,
    "Content-Type": "application/json"
}

# Requisição ao endpoint
try:
    response = requests.post(endpoint_url, headers=headers, json=input_data)
    response.raise_for_status()  # Levanta exceção para erros HTTP
    result = response.json()
    print("Resposta do modelo:", result)
except requests.exceptions.RequestException as e:
    print("Erro ao chamar o endpoint:", e)
