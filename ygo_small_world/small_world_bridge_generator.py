"""
Module: small_world_bridge_generator

Part of the YGO-small-world project, this module is for identifying optimal 'Small World' bridges in Yu-Gi-Oh! decks.
It includes functionalities to generate adjacency matrices, calculate bridge scores, and find effective bridges for specific decks.

Key Functions:
- ydk_to_df_adjacency_matrix: Generates adjacency matrix from YDK files.
- calculate_bridge_score: Computes bridge score for potential bridge cards.
- find_best_bridges: Determines optimal bridges for a given deck.

Note: Understanding of Yu-Gi-Oh! card properties and Small World mechanics is essential.
"""

import json
from pathlib import Path
from functools import cache

import pandas as pd
import numpy as np

from ygo_small_world import fetch_card_data as fcd

def sub_df(df: pd.DataFrame, column_values: list, column_name: str) -> pd.DataFrame:
    """
    Utility function. Creates a subset of the given DataFrame based on specified values in a particular column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame from which the subset will be extracted.
        column_values (list): A list of values to match against the specified column to filter rows.
        column_name (str): The name of the column in which to look for the specified values.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows where the specified column contains any of the values in 'column_values'.
    """
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' is not a valid column in the DataFrame.")

    if not pd.Series(column_values).isin(df[column_name]).all():
        raise ValueError("Not all values are in column.")

    mask = df[column_name].isin(column_values)
    return df.loc[mask].copy()

def load_cards() -> pd.DataFrame:
    """
    Loads a DataFrame containing information about all cards from a JSON file. 
    The JSON file should contain data for all cards.

    Returns:
        pd.DataFrame: A DataFrame containing information about all cards, 
                      including their ID, name, type, attribute, level, attack, and defense.
    """
    current_dir = Path.cwd()
    cardinfo_path = current_dir / "data" / "cardinfo.json"

    # Pull card data if it doesn't exist
    if not cardinfo_path.exists():
        print("Card data missing, fetching card data.")
        fcd.fetch_card_data()

    # Load the contents of cardinfo.json
    with open(cardinfo_path, 'r', encoding='utf-8') as file_path:
        json_all_cards = json.load(file_path)

    column_rename_map = {'type': 'category', 'race': 'type'} # rename columns to be less confusing
    df_all_cards = pd.DataFrame(json_all_cards['data']).rename(columns=column_rename_map)

    return df_all_cards

@cache
def load_main_monsters() -> pd.DataFrame:
    """
    Filters a DataFrame containing information about all cards to only main deck monster cards. 

    Returns:
        pd.DataFrame: A DataFrame containing information about all main deck monsters, 
                      including their ID, name, type, attribute, level, attack, and defense.
    """

    df_all_cards = load_cards()
    #only keep main deck monsters
    main_monster_frame_types = ['effect', 'normal', 'effect_pendulum',
                                'ritual', 'normal_pendulum', 'ritual_pendulum']
    df_main_monsters = sub_df(df_all_cards, main_monster_frame_types, 'frameType').reset_index(drop=True)

    #filter relevant columns
    relevent_columns = ['id', 'name', 'type', 'attribute', 'level', 'atk', 'def']
    df_main_monsters = df_main_monsters[relevent_columns]

    return df_main_monsters

def monster_names_to_df(card_names: list[str]) -> pd.DataFrame:
    """
    Converts a list of monster card names into a DataFrame containing details of those monsters.

    Parameters:
        card_names (list): List of monster card names as strings.

    Returns:
        pd.DataFrame: A DataFrame containing the information of the specified monster cards.
    """
    main_monsters = load_main_monsters()
    return sub_df(main_monsters, card_names, 'name')

# READ YDK FILES

def ydk_to_card_ids(ydk_file: str) -> list[int]:
    """
    Extracts card IDs from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file.

    Returns:
        list: A list of card IDs as integers.
    """
    card_ids = []
    with open(ydk_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            try:
                card_id = int(line)
            except ValueError:
                pass
            else:
                card_ids.append(card_id)
    return card_ids

def ydk_to_monster_names(ydk_file: str) -> list[str]:
    """
    Extracts the names of main deck monsters from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file.

    Returns:
        list: A list of names of main deck monsters present in the ydk file.
    """
    card_ids = ydk_to_card_ids(ydk_file)
    main_monsters = load_main_monsters()
    df_monsters = sub_df(main_monsters, card_ids, 'id')
    monster_names = df_monsters['name'].tolist()
    return monster_names

#### ADJACENCY MATRIX GENERATION ####

def df_to_adjacency_matrix(df_cards: pd.DataFrame, squared: bool = False) -> np.ndarray:
    """
    Creates an adjacency matrix based on Small World connections for a given DataFrame of cards.
    Two cards are considered adjacent if they have exactly one property in common from the following attributes: type, attribute, level, attack, or defense.

    Parameters:
        df_cards (pd.DataFrame): DataFrame containing information about the cards.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        np.array: An adjacency matrix representing the connections between cards.
    """
    required_columns = ['type', 'attribute', 'level', 'atk', 'def']
    if not all(column in df_cards.columns for column in required_columns):
        raise ValueError("DataFrame must have columns: 'type', 'attribute', 'level', 'atk', 'def'")

    # Extract relevant columns and convert to numpy array
    card_attributes = df_cards[required_columns].to_numpy()

    # Broadcasting to compare each card with every other card
    # This creates a 3D array where the third dimension is the attribute comparison between cards
    comparisons = card_attributes[:, np.newaxis, :] == card_attributes

    # Sum along the last axis to count the number of similarities between each pair of cards
    similarity_count = comparisons.sum(axis=2)

    # Create the adjacency matrix where exactly one attribute matches
    adjacency_matrix = (similarity_count == 1).astype(int)

    if squared:
        adjacency_matrix = adjacency_matrix @ adjacency_matrix

    return adjacency_matrix

def names_to_adjacency_matrix(card_names: list[str], squared: bool = False) -> np.ndarray:
    """
    Creates an adjacency matrix based on Small World connections for a list of monster card names.

    Parameters:
        card_names (list): List of monster card names.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        ndarray: An adjacency matrix representing the connections between the named cards.
    """
    df_cards = monster_names_to_df(card_names)
    adjacency_matrix = df_to_adjacency_matrix(df_cards, squared=squared)
    return adjacency_matrix

def names_to_labeled_adjacency_matrix(card_names: list[str], squared: bool = False) -> pd.DataFrame:
    """
    Creates a labeled adjacency matrix DataFrame based on Small World connections for a given list of monster names.

    Parameters:
        card_names (list): List of monster names.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        pd.DataFrame: A labeled adjacency matrix with both row and column names corresponding to the monster names.
    """
    adjacency_matrix = names_to_adjacency_matrix(card_names, squared=squared)
    return pd.DataFrame(adjacency_matrix, index=card_names, columns=card_names)

def ydk_to_adjacency_matrix(ydk_file: str, squared: bool = False) -> pd.DataFrame:
    """
    Creates an adjacency matrix DataFrame based on Small World connections from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file containing the deck information.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        pd.DataFrame: A labeled adjacency matrix with both row and column names corresponding to the names of monsters in the ydk file.
    """
    card_names = ydk_to_monster_names(ydk_file)
    return names_to_adjacency_matrix(card_names, squared=squared)

def ydk_to_labeled_adjacency_matrix(ydk_file: str, squared: bool = False) -> pd.DataFrame:
    """
    Creates a labeled adjacency matrix DataFrame based on Small World connections from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file containing the deck information.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        pd.DataFrame: A labeled adjacency matrix with both row and column names corresponding to the names of monsters in the ydk file.
    """
    card_names = ydk_to_monster_names(ydk_file)
    return names_to_labeled_adjacency_matrix(card_names, squared=squared)

#### BRIDGE FINDING ####

@cache
def calculate_all_cards_adjacency_matrix() -> np.ndarray:
    """
    Returns:
    - np.ndarray: Calculates adjacency matrix of all main deck monsters.
    """
    main_monsters = load_main_monsters()
    return df_to_adjacency_matrix(main_monsters)

def filter_main_monsters(required_target_names: list[str]) -> pd.DataFrame:
    """
    Filters and returns main monsters that connect to all required target names.

    Parameters:
    - required_target_names (list[str]): List of target names that the main monsters must be connected to.

    Returns:
    - pd.DataFrame: A dataframe of main monsters that have connections to all the required target names.
    """
    # Load main monsters and adjacency matrix
    main_monsters = load_main_monsters()
    all_cards_adjacency_matrix = calculate_all_cards_adjacency_matrix()

    required_indices = sub_df(main_monsters, required_target_names, 'name').index

    # Calculate connections to required targets
    num_connections = all_cards_adjacency_matrix[required_indices, :].sum(axis=0)

    # Filter main monsters connected to all required targets
    required_target_mask = num_connections == len(required_target_names)
    return main_monsters[required_target_mask].copy()

def calculate_bridge_matrix(df_deck, df_bridges):
    """
    Constructs a bridge matrix from a given deck and bridge dataframes.

    The bridge matrix is a subset of the full adjacency matrix of all cards, representing 
    connections between monsters in the deck and those satisfying the required connections.

    Parameters:
    - df_deck (pd.DataFrame): Dataframe representing the deck, with card indices.
    - df_bridges (pd.DataFrame): Dataframe representing bridge monsters, with card indices.

    Returns:
    - np.ndarray: The bridge matrix indicating connections between deck monsters and bridge monsters.
    """
    # Calculate full adjacency matrix
    all_cards_adjacency_matrix = calculate_all_cards_adjacency_matrix()

    # Get indices of monsters that satisfy all required connections
    bridge_indices = df_bridges.index
    deck_indices = df_deck.index

    # Construct bridge matrix
    return all_cards_adjacency_matrix[deck_indices,:][:,bridge_indices]


def calculate_bridge_scores(deck_monster_names: list[str], bridge_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates bridge scores for a deck of cards. The score is the number of non-zero entries in the squared 
    adjacency matrix (representing deck connections) adjusted by the bridge matrix, normalized by the square of 
    (number of cards in the deck + 1).

    Parameters:
    - deck_monster_names (list[str]): Names of monsters in the deck.
    - bridge_matrix (np.ndarray): Bridge matrix for the deck with the required connections.

    Returns:
    - np.ndarray: Array of calculated bridge scores for each monster card.

    Raises:
    - ValueError: If the dimensions of the bridge matrix do not match the expected size.
    """
    deck_adjacency_matrix = names_to_adjacency_matrix(deck_monster_names)
    deck_adjacency_matrix_squared = deck_adjacency_matrix @ deck_adjacency_matrix

    num_deck_cards = len(deck_monster_names)
    if deck_adjacency_matrix_squared.shape[0] != num_deck_cards:
        raise ValueError("Mismatch in dimensions between the deck adjacency matrix and bridge matrix.")

    i,j = np.mgrid[:num_deck_cards, :num_deck_cards]
    outer_product_tensor = bridge_matrix[i] * bridge_matrix[j] #outer product of connection vectors
    deck_connection_tensor = outer_product_tensor + deck_adjacency_matrix_squared[:, :, np.newaxis] #A^2 + x(x.T) for all connection vectors x
    deck_connectivity = deck_connection_tensor.astype(bool).sum(axis=(0,1)) #number of non-zero elements in each slice

    bridge_connection_matrix = deck_adjacency_matrix @ bridge_matrix
    bridge_connectivity = bridge_connection_matrix.astype(bool).sum(axis=0) #num non-zero elements in each row

    # Formula for bridge score derived from block matrix multiplication.
    bridge_score = (deck_connectivity + 2*bridge_connectivity + 1)/((num_deck_cards+1)**2)
    return bridge_score

def assemble_df_bridges(df_bridges: pd.DataFrame, number_of_connections: list[int], bridge_score: list[float]) -> pd.DataFrame:
    """
    Adds 'number_of_connections' and 'bridge_score' to 'df_bridges', filters out entries with no connections, 
    and sorts the dataframe. The final dataframe is sorted by bridge score (descending), number of connections 
    (descending), and name (ascending).

    Parameters:
    - df_bridges (pd.DataFrame): Dataframe with bridge data.
    - number_of_connections (list[int]): Connection counts for each bridge.
    - bridge_score (list[float]): Scores for each bridge.

    Returns:
    - pd.DataFrame: Updated and sorted dataframe.
    """
    if len(df_bridges) != len(number_of_connections) or len(df_bridges) != len(bridge_score):
        raise ValueError("Length of 'number_of_connections' and 'bridge_score' must match the length of 'df_bridges'.")

    df_bridges['number_of_connections'] = number_of_connections
    df_bridges['bridge_score'] = bridge_score

    # Filter out cards with 0 connections.
    df_bridges = df_bridges[df_bridges['number_of_connections'] > 0]

    # Reorder columns.
    df_bridges = df_bridges[['bridge_score', 'number_of_connections', 'name', 'type', 'attribute', 'level', 'atk', 'def']]

    # Reorder rows.
    df_bridges = df_bridges.sort_values(by=['bridge_score', 'number_of_connections', 'name'], ascending=[False, False, True]).reset_index(drop=True) 
    return df_bridges

def find_best_bridges(deck_monster_names: list[str], required_target_names: list[str] = None, top: int = None) -> pd.DataFrame:
    """
    Identifies the best bridges (monsters) that connect the most cards in the deck via Small World
    and connect to all the required targets.

    Parameters:
    - deck_monster_names (list): A list of monster names in the deck.
    - required_target_names (list, optional): A list of monster names that must be connected to the bridges via Small World.
        Default is an empty list.
    top (int, optional): The number of top bridges to return.

    Returns:
        DataFrame: A Pandas DataFrame containing details of the best bridges including bridge score, number of connections,
          name, type, attribute, level, attack, and defense. If no bridges meet the requirements, prints a message and returns None.
    """
    main_monsters = load_main_monsters()

    if required_target_names:
        # Union names so required_target_names is a subset of deck_monster_names
        deck_monster_names = list(set(deck_monster_names) | set(required_target_names))
        # Filter main_monsters to only include cards that connect with all cards in required_target_names.
        df_bridges = filter_main_monsters(required_target_names)
        if len(df_bridges)==0:
            raise ValueError('There are no monsters that bridge all required targets.')
    else:
        deck_monster_names = list(set(deck_monster_names))
        df_bridges = main_monsters

    df_deck = sub_df(main_monsters, deck_monster_names, 'name')
    # bridge_matrix is subset of adjacency matrix corresponding to (deck monsters) by (monsters with connections to the required cards)
    bridge_matrix = calculate_bridge_matrix(df_deck, df_bridges)

    # One array entry for each monster in df_bridges, with entry corresponding to number of connections.
    number_of_connections = bridge_matrix.sum(axis=0)

    # Calculate bridge score = num non-zero entries in square of adjacency matrix if bridge was included divided by square of num cards in deck + 1.
    bridge_score = calculate_bridge_scores(deck_monster_names, bridge_matrix)

    #assemble df
    return assemble_df_bridges(df_bridges, number_of_connections, bridge_score).head(top)

def find_best_bridges_from_ydk(ydk_file: str, top: int = None) -> pd.DataFrame:
    """
    Identifies the best bridges that connect the most cards in the deck from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
    - ydk_file (str): Path to the ydk file of the deck.
    - top (int, optional): The number of top bridges to return.

    Returns:
    - DataFrame: A Pandas DataFrame containing details of the best bridges. The same as returned by `find_best_bridges`.
    """
    deck_monster_names = ydk_to_monster_names(ydk_file)
    df_bridges = find_best_bridges(deck_monster_names, top=top)
    return df_bridges

### MISC FUNCTIONS FOR STATISTICS ###

def top_bridges(reverse: bool = False, num: int = 10, ) -> pd.DataFrame:
    """Returns the top bridges of all cards.
    Optional arguments: reverse to return bottom bridges, num to specify the number of bridges."""
    total_connections = calculate_all_cards_adjacency_matrix().sum(axis=0)
    main_monsters = load_main_monsters().copy()
    main_monsters.insert(2, 'total connections', total_connections)
    return main_monsters.sort_values(by=['total connections'], ascending=reverse).head(num)

if __name__ == "__main__":
    print(f"The top bridges are: {top_bridges()}.\n")
    print(f"The bottom bridges are: {top_bridges(reverse=True)}.\n")
    print(f"The total number of main deck monster cards is {len(load_main_monsters())}")
