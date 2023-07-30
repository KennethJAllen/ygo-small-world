import requests
import json 

HOUSING_URL = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'

def fetch_card_data():
    res =  requests.get(HOUSING_URL)
    response = json.loads(res.text)
    with open('cardinfo.json', 'w') as f:
        json.dump(response, f)

fetch_card_data()