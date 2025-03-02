"""Updates card data."""
import json
import requests
import pandas as pd
from pyprojroot import here
from ygo_small_world import utils

def update_card_data() -> None:
    """Updates card data in DataFrame pickle."""
    card_info = _fetch_card_data()
    df_all_cards = _card_json_to_df(card_info)
    df_main_monsters = _filter_card_df(df_all_cards)

    # Save to pikle
    output_path = here() / "data" / "cardinfo.pkl"
    df_main_monsters.to_pickle(output_path)

def _fetch_card_data() -> dict:
    """
    Retrieves card data from the Yu-Gi-Oh! API
    Returns dictionary of card data
    """
    housing_url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'
    res =  requests.get(housing_url, timeout=10)
    card_info = json.loads(res.text)
    return card_info

def _card_json_to_df(card_info: dict) -> pd.DataFrame:
    """Converts card info json to DataFrame"""
    column_rename_map = {'type': 'category', 'race': 'type'} # rename columns to be less confusing
    df_all_cards = pd.DataFrame(card_info['data']).rename(columns=column_rename_map)
    return df_all_cards

def _filter_card_df(all_cards_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters card info DataFrame to only main deck monster cards.
    These are the only cards relevant for the card Small World.
    """
    # Only keep main deck monsters
    main_monster_frame_types = ['effect', 'normal', 'effect_pendulum',
                                'ritual', 'normal_pendulum', 'ritual_pendulum']
    df_main_monsters = utils.sub_df(all_cards_df, main_monster_frame_types, 'frameType').reset_index(drop=True)

    # Process card image url column
    df_main_monsters['card_images'] = df_main_monsters['card_images'].apply(lambda x: x[0]['image_url_cropped'])

    # Only keep relevant columns
    relevent_columns = ['id', 'name', 'type', 'attribute', 'level', 'atk', 'def', 'card_images']
    df_main_monsters = df_main_monsters[relevent_columns]
    df_main_monsters.set_index('name')

    return df_main_monsters

if __name__ == "__main__":
    update_card_data()
