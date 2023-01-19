import json
import pandas as pd
import numpy as np

def sub_df(df, column_values, column_name):
    #creates subset of dataframe consisting of rows with column values in column
    mask = df[column_name].apply(lambda x: any(value for value in column_values if value in x))
    return df[mask]

def load_main_monsters():
    #with open('cardinfo.php.json') as file_path:
    with open('cardinfo.json') as file_path:
        cards_json = json.load(file_path)
    all_cards = pd.DataFrame(cards_json['data'])
    all_cards = all_cards.rename(columns={'type': 'card_type','race':'type'})

    main_monster_card_types = ['Effect Monster',
                                'Normal Monster',
                                'Flip Effect Monster',
                                'Union Effect Monster',
                                'Pendulum Effect Monster',
                                'Tuner Monster',
                                'Gemini Monster',
                                'Normal Tuner Monster',
                                'Spirit Monster',
                                'Ritual Effect Monster',
                                'Ritual Monster',
                                'Toon Monster',
                                'Pendulum Normal Monster',
                                'Pendulum Tuner Effect Monster',
                                'Pendulum Effect Ritual Monster',
                                'Pendulum Flip Effect Monster']
    main_monsters = sub_df(all_cards, main_monster_card_types, 'card_type').reset_index(drop=True)
    main_monsters = main_monsters[['name','type','attribute','level','atk','def']] #keep only relevant columns
    return main_monsters

def create_sw_adjacency_matrix(deck_mosters):
    #creates adjacency matrix corresponding to Small World connections
    #two cards are considered adjacent if they have exactly one type, attribute, level, atk, or def in common
    deck_monsters_array = deck_mosters.to_numpy()
    num_cards = len(deck_mosters)
    adjacency_matrix = np.zeros((num_cards,num_cards))
    for i in range(num_cards):
        card_similarities = deck_monsters_array==deck_monsters_array[i]
        similarity_measure = card_similarities.astype(int).sum(axis=1)
        adjacency_matrix[:,i] = (similarity_measure==1).astype(int) #indicates where there is exactly one similarity
    return adjacency_matrix