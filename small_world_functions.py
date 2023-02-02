import json
import pandas as pd
import numpy as np

def sub_df(df, column_values, column_name):
    #creates subset of dataframe consisting of rows with column_values in column
    df = df.copy()
    mask = df[column_name].apply(lambda x: any(value for value in column_values if value == x))
    return df[mask]

def load_main_monsters():
    #loads dataframe of all main deck monsters
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
    df_main_monsters = sub_df(df_all_cards, main_monster_card_category, 'category').reset_index(drop=True) #only keep main deck monsters
    df_main_monsters = df_main_monsters[['id', 'name','type','attribute','level','atk','def']] #keep only relevant columns
    return df_main_monsters

MAIN_MONSTERS = load_main_monsters()

def monster_names_to_df(card_names):
    #converts list of monster names into a dataframe of those monsters
    df_cards = sub_df(MAIN_MONSTERS, card_names, 'name')
    return df_cards

#### READING YDK FILES ####

def ydk_to_card_ids(ydk_file):
    #convers a ydk file to card ids
    card_ids = []
    with open(ydk_file) as f:
        lines = f.readlines()
        for line in lines:
            try:
                id = int(line)
            except:
                pass
            else:
                card_ids.append(id)
    return card_ids

def ydk_to_monster_names(ydk_file):
    #input: ydk file, which consists of card IDs
    #output: list of names of main deck monsters in ydk file
    card_ids = ydk_to_card_ids(ydk_file)
    df_monsters = sub_df(MAIN_MONSTERS, card_ids, 'id')
    monster_names = df_monsters['name'].tolist()
    return monster_names

#### ADJACENCY MATRICES GENERATION ####

def df_to_adjacency_matrix(df_cards, squared=False):
    #creates adjacency array corresponding to Small World connections
    #two cards are considered adjacent if they have exactly one type, attribute, level, atk, or def in common
    df_cards = df_cards[['type','attribute','level','atk','def']]
    array_cards = df_cards.to_numpy()
    num_cards = len(df_cards)
    adjacency_matrix = np.zeros((num_cards,num_cards))
    for i in range(num_cards):
        card_similarities = array_cards==array_cards[i]
        similarity_measure = card_similarities.astype(int).sum(axis=1)
        adjacency_matrix[:,i] = (similarity_measure==1).astype(int) #indicates where there is exactly one similarity
    if squared==True:
        adjacency_matrix = np.linalg.matrix_power(adjacency_matrix, 2)
    return adjacency_matrix

SW_ADJACENCY_MATRIX = df_to_adjacency_matrix(MAIN_MONSTERS) #small world adjacency array of all cards

def names_to_labeled_adjacency_matrix(card_names, squared=False):
    #input: list of monster names. Optional parameter to square resulting matrix
    #output: adjacency matrix dataframe
    df_cards = monster_names_to_df(card_names)
    adjacency_matrix = df_to_adjacency_matrix(df_cards)
    if squared==True:
        adjacency_matrix = np.linalg.matrix_power(adjacency_matrix, 2)
    df_adjacency_matrix = pd.DataFrame(adjacency_matrix, index=card_names, columns=card_names)
    return df_adjacency_matrix

def ydk_to_labeled_adjacency_matrix(ydk_file, squared=False):
    #input: ydk file of deck. Optional parameter to square resulting matrix
    #output: adjacency matrix dataframe
    card_names = ydk_to_monster_names(ydk_file)
    df_adjacency_matrix = names_to_labeled_adjacency_matrix(card_names, squared=squared)
    return df_adjacency_matrix

#### BRIDGE FINDING ####

def find_best_bridges(deck_monster_names, required_target_names=[]):
    #inputs: list of monster names and list of monsters that are required to connect with the small world bridges
    #output: The bridges that connect the most cards in your deck and connect with all the required targets
    deck_monster_names = list(set(deck_monster_names) | set(required_target_names)) #union names so required_target_names is a subset of deck_monster_names
    deck_indices = sub_df(MAIN_MONSTERS, deck_monster_names, 'name').index
    required_indices = sub_df(MAIN_MONSTERS, required_target_names, 'name').index #indices of required targets

    num_required_targets = len(required_target_names) #number of cards required to connect with one bridge

    required_monster_matrix = SW_ADJACENCY_MATRIX[required_indices,:] #array corresponding to required connection monsters by all monsters
    num_bridges_to_required_cards = required_monster_matrix.sum(axis=0) #number of required connections satisfied by all monsters
    required_bridge_mask = num_bridges_to_required_cards==num_required_targets
    df_bridges = MAIN_MONSTERS[required_bridge_mask].copy() #data frame of monsters connecting all required targets
    required_bridge_indices = df_bridges.index #indices of monsters that satisfy all required connections
    if len(df_bridges)==0:
        print('There are no monsters that bridge all required targets.')
        return

    #subset of adjacency matrix corresponding to (deck monsters) by (monsters with connections to the required cards)
    bridge_matrix = SW_ADJACENCY_MATRIX[deck_indices,:][:,required_bridge_indices]

    num_deck_bridges = bridge_matrix.sum(axis=0)
    df_bridges['number_of_connections'] = num_deck_bridges
    df_bridges = df_bridges[df_bridges['number_of_connections'] > 0]
    df_bridges = df_bridges[['number_of_connections', 'name', 'type', 'attribute', 'level', 'atk', 'def']] #reorder columns
    df_bridges = df_bridges.sort_values(by=['number_of_connections','name'], ascending=[False, True]).reset_index(drop=True) #reorder rows
    return df_bridges

def find_best_bridges_from_ydk(ydk_file):
    #inputs: ydk file of deck
    #output: The bridges that connect the most cards in your deck
    deck_monster_names = ydk_to_monster_names(ydk_file)
    df_bridges = find_best_bridges(deck_monster_names)
    return df_bridges

#### IMAGE GENERATION ####