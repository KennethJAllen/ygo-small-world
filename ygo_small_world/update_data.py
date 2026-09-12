"""Updates card data."""
import os
from math import isfinite
from tempfile import NamedTemporaryFile
import requests
import pandas as pd
from ygo_small_world import utils
from ygo_small_world.data_paths import writable_card_path

def update_card_data() -> None:
    """Updates card data in DataFrame pickle."""
    card_info = _fetch_card_data()
    df_all_cards = _card_json_to_df(card_info)
    df_main_monsters = _filter_card_df(df_all_cards)
    _validate_card_df(df_main_monsters)
    df_main_monsters = df_main_monsters.sort_values('id').reset_index(drop=True)

    output_path = writable_card_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(dir=output_path.parent, suffix='.pkl', delete=False) as temporary:
            temporary_path = temporary.name
            df_main_monsters.to_pickle(temporary)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _validate_card_df(cards: pd.DataFrame) -> None:
    """Reject incomplete API data before replacing a working database."""
    if cards.empty or cards.isna().any().any():
        raise ValueError('Card data must contain complete monster properties.')
    for column in ('id', 'level', 'atk', 'def'):
        values = pd.to_numeric(cards[column], errors='raise')
        if not values.map(isfinite).all():
            raise ValueError(f'Card data contains invalid {column} values.')
        cards[column] = values
    if cards['id'].duplicated().any():
        raise ValueError('Card data must contain unique IDs.')
    for column in ('name', 'type', 'attribute', 'img_url'):
        if not cards[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f'Card data contains invalid {column} values.')

def _fetch_card_data() -> dict:
    """
    Retrieves card data from the Yu-Gi-Oh! API
    Returns dictionary of card data
    """
    url = 'https://db.ygoprodeck.com/api/v7/cardinfo.php'
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    card_info = res.json()
    if not isinstance(card_info, dict) or not isinstance(card_info.get('data'), list) or not card_info['data']:
        raise ValueError('Card API response must contain a nonempty data list.')
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
    df_main_monsters['img_url'] = df_main_monsters['card_images'].apply(lambda x: x[0]['image_url_cropped'])

    # Only keep relevant columns
    relevent_columns = ['id', 'name', 'type', 'attribute', 'level', 'atk', 'def', 'img_url']
    df_main_monsters = df_main_monsters[relevent_columns]
    return df_main_monsters

if __name__ == "__main__":
    update_card_data()
