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
        json_all_cards = json.load(file_path)
    df_all_cards = pd.DataFrame(json_all_cards['data'])
    df_all_cards = df_all_cards.rename(columns={'type': 'category','race':'type'})

    main_monster_card_category = ['Effect Monster',
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
    df_main_monsters = sub_df(df_all_cards, main_monster_card_category, 'category').reset_index(drop=True)
    df_main_monsters = df_main_monsters[['name','type','attribute','level','atk','def']] #keep only relevant columns
    return df_main_monsters

def create_sw_adjacency_matrix(df_cards):
    #creates adjacency matrix corresponding to Small World connections
    #two cards are considered adjacent if they have exactly one type, attribute, level, atk, or def in common
    array_cards = df_cards.to_numpy()
    num_cards = len(df_cards)
    adjacency_matrix = np.zeros((num_cards,num_cards))
    for i in range(num_cards):
        card_similarities = array_cards==array_cards[i]
        similarity_measure = card_similarities.astype(int).sum(axis=1)
        adjacency_matrix[:,i] = (similarity_measure==1).astype(int) #indicates where there is exactly one similarity
    return adjacency_matrix