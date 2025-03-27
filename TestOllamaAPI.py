import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

data = {
    "model": "llama3.2:latest",
    "messages": [{"role": "user", "content": "Dis-moi un fun fact sur l'IA"}]
}

response = requests.post(OLLAMA_URL, json=data)

# Vérifier le code de statut et afficher le contenu
if response.status_code == 200:
    try:
        print(response.json())
    except ValueError as e:
        print(f"Erreur de décodage JSON : {e}")
        print("Contenu de la réponse :", response.text)
else:
    print(f"Erreur {response.status_code}: {response.text}")
