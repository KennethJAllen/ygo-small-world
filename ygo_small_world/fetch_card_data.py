import requests
import json
import os

HOUSING_URL = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'

def fetch_card_data():
    res =  requests.get(HOUSING_URL)
    response = json.loads(res.text)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cardinfo_path = os.path.join(current_dir, "cardinfo.json")

    # Load the contents of cardinfo.json
    with open(cardinfo_path, 'w') as f:
        json.dump(response, f)

fetch_card_data()