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
from pathlib import Path
import pandas as pd
import numpy as np
from pyprojroot import here
from ygo_small_world import utils
from ygo_small_world.update_data import update_card_data

def main():
    all_cards = AllCards()
    print(all_cards.top_bridges(20))

def df_to_adjacency_matrix(df_cards: pd.DataFrame) -> np.ndarray:
    """
    Creates an adjacency matrix based on Small World connections for a given DataFrame of cards.
    Two cards are considered adjacent if they have exactly one property in common from the following attributes: type, attribute, level, attack, or defense.

    Parameters:
        df_cards (pd.DataFrame): DataFrame containing information about the cards.

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

    return adjacency_matrix

class AllCards:
    """Contains data for main deck monster cards relevant for Small World"""
    def __init__(self):
        self._df: pd.DataFrame = self._load_cards()
        self._adjacency_matrix: np.ndarray = df_to_adjacency_matrix(self._df)

    def get_df(self):
        """
        DataFrame containing information about cards.
        Including their ID, name, type, attribute, level, attack, defense, and card_images url.
        """
        return self._df

    def get_adjacency_matrix(self):
        """The adjacency matrix of the cards. Two cards are considered connected if they have a connection via Small World."""
        return self._adjacency_matrix

    def get_labeled_adjacency_matrix(self):
        """Returns adjacency matrix labeled with card names."""
        card_names = self._df['name'].tolist()
        return pd.DataFrame(self._adjacency_matrix, index=card_names, columns=card_names)

    def top_bridges(self, num: int = 10, reverse: bool = False) -> pd.DataFrame:
        """Returns the top bridges of all cards.
        Optional arguments: reverse to return bottom bridges, num to specify the number of bridges."""
        total_connections = self._adjacency_matrix.sum(axis=0)
        self._df.insert(2, 'num_connections', total_connections)
        return self._df.sort_values(by=['num_connections'], ascending=reverse).head(num)

    def filter_required_targets(self, required_target_ids: list[int]) -> pd.DataFrame:
        """Filters and returns cards that connect to all required target names"""
        required_indices = utils.sub_df(self._df, required_target_ids, 'id').index

        # Calculate number of connections to required targets
        num_connections = self._adjacency_matrix[required_indices, :].sum(axis=0)

        # Filter main monsters connected to all required targets
        required_target_mask = num_connections == len(required_target_ids)
        return self._df[required_target_mask].copy()

    def _load_cards(self) -> pd.DataFrame:
        """
        Loads a DataFrame containing information about all main monster cards.
        Including their ID, name, type, attribute, level, attack, defense, and card_images url
        """
        root_dir = here()
        cardinfo_path = root_dir / "data" / "cardinfo.pkl"

        # Pull card data if it doesn't exist
        if not cardinfo_path.exists():
            print("Card data missing, fetching card data.")
            update_card_data()

        # Load the contents of card data
        df_all_cards = pd.read_pickle(cardinfo_path)

        return df_all_cards

class Deck:
    """Contains data for deck relevant for Small World"""
    def __init__(self, deck_ids: list[int], all_cards: AllCards):
        self._deck_ids = deck_ids
        self._all_cards = all_cards
        self._df = utils.sub_df(all_cards.get_df(), self._deck_ids, 'id')
        self._adjacency_matrix = None
        self._squared_adjacency_matrix = None

    def get_df(self):
        """Returns dataframe of deck."""
        return self._df

    def get_adjacency_matrix(self):
        """Returns the Small World adjacency matrix for deck.
        Generates it if it does not exist."""
        if self._adjacency_matrix is None:
            deck_indices = self._df.index
            self._adjacency_matrix = self._all_cards.get_adjacency_matrix()[deck_indices,:][:,deck_indices]
        return self._adjacency_matrix

    def get_squared_adjacency_matrix(self):
        """Returns the square of the Small World adjacency matrix for deck.
        Generates it if it does not exist."""
        if self._adjacency_matrix is None:
            self.get_adjacency_matrix()
        if self._squared_adjacency_matrix is None:
            self._squared_adjacency_matrix = self._adjacency_matrix @ self._adjacency_matrix
        return self._squared_adjacency_matrix

    def get_labeled_adjacency_matrix(self):
        """Returns adjacency matrix labeled with card names."""
        card_names = self._df['names'].to_list()
        return pd.DataFrame(self._adjacency_matrix, index=card_names, columns=card_names)

def ydk_to_card_ids(ydk_path: Path) -> list[int]:
    """
    Extracts card IDs from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file.

    Returns:
        list: A list of card IDs as integers.
    """
    card_ids = []
    with open(ydk_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            try:
                card_id = int(line)
            except ValueError:
                pass
            else:
                card_ids.append(card_id)
    return card_ids

def get_bridge_matrix(deck: Deck, card_pool: Deck):
    """
    Constructs a bridge matrix from a given deck and card pool dataframes.

    The bridge matrix is a subset of the full adjacency matrix of all cards, representing 
    connections between monsters in the deck and those satisfying the required connections.

    If there are m cards in total and n cards in the deck, the result should be n x m

    
    Returns:
    - np.ndarray: The bridge matrix indicating connections between deck monsters and bridge monsters.
    """
    # Calculate full adjacency matrix
    pool_adjacency_matrix = card_pool.get_adjacency_matrix()

    # Get indices of monsters that satisfy all required connections
    pool_indices = card_pool.get_df().index
    deck_indices = deck.get_df().index

    # Construct bridge matrix
    return pool_adjacency_matrix[deck_indices,:][:,pool_indices]


#### WORK IN PROGRESS


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
    deck_adjacency_matrix = card_ids_to_adjacency_matrix(deck_monster_names)
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
    main_monsters = load_cards()

    if required_target_names:
        # Union names so required_target_names is a subset of deck_monster_names
        deck_monster_names = list(set(deck_monster_names) | set(required_target_names))
        # Filter main_monsters to only include cards that connect with all cards in required_target_names.
        df_bridges = filter_required_targets(required_target_names)
        if len(df_bridges)==0:
            raise ValueError('There are no monsters that bridge all required targets.')
    else:
        deck_monster_names = list(set(deck_monster_names))
        df_bridges = main_monsters

    df_deck = utils.sub_df(main_monsters, deck_monster_names, 'name')
    # bridge_matrix is subset of adjacency matrix corresponding to (deck monsters) by (monsters with connections to the required cards)
    bridge_matrix = get_bridge_matrix(df_deck, df_bridges)

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


if __name__ == "__main__":
    main()
