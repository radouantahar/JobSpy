import requests
from bs4 import BeautifulSoup

def get_train_travel_time_trainline(origin, destination):
    url = f"https://api.trainline.eu/v1/search?from={origin}&to={destination}"
    
    response = requests.get(url)
    data = response.json()
    
    if 'results' in data and len(data['results']) > 0:
        duration = data['results'][0]['duration']  # Durée en minutes
        return duration
    else:
        return None

def scrape_train_schedule(origin, destination):
    url = f"https://www.example.com/train-schedule?from={origin}&to={destination}"  # Remplacez par l'URL réelle
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Exemple de recherche d'éléments dans le HTML
    # Vous devrez adapter cela en fonction de la structure du site
    times = soup.find_all('div', class_='travel-time')  # Remplacez par la classe réelle
    for time in times:
        print(time.text)

# Exemple d'utilisation
origin = "Paris"
destination = "Lyon"
travel_time = get_train_travel_time_trainline(origin, destination)

if travel_time:
    print(f"Le temps de transport en train entre {origin} et {destination} est d'environ {travel_time} minutes.")
else:
    print("Erreur lors de la récupération des données.")

scrape_train_schedule("Paris", "Lyon")
