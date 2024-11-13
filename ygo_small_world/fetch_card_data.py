"""Updates card data."""

import json
from pathlib import Path
import requests

def fetch_card_data() -> None:
    """
    Retrieves card data from housing_url and saves it as 'cardinfo.json' in the script's directory.

    Uses a 10-second timeout for the HTTP request and writes the received JSON data to the file.
    """
    housing_url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'
    res =  requests.get(housing_url, timeout=10)
    response = json.loads(res.text)
    current_dir = Path.cwd()
    cardinfo_path = current_dir / "data" / "cardinfo.json"

    # Load the contents of cardinfo.json
    with open(cardinfo_path, 'w', encoding='utf-8') as f:
        json.dump(response, f)

if __name__ == "__main__":
    fetch_card_data()
