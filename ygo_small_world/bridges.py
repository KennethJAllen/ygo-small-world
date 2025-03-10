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
import networkx as nx
from networkx.classes.graph import Graph
from pyprojroot import here
from ygo_small_world import utils
from ygo_small_world.update_data import update_card_data


class AllCards:
    """Contains data for main deck monster cards relevant for Small World"""
    def __init__(self):
        self._df: pd.DataFrame = self._load_cards()
        self._adjacency_matrix: np.ndarray = self._calculate_to_adjacency_matrix()

    def __len__(self):
        return len(self._df)

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

    def filter_required_targets(self, required_target_ids: list[int]) -> pd.Series:
        """Filters and returns cards that connect to all required target names"""
        required_indices = utils.sub_df(self._df, required_target_ids, 'id').index

        # Calculate number of connections to required targets
        num_connections = self._adjacency_matrix[required_indices, :].sum(axis=0)

        # Filter main monsters connected to all required targets
        required_target_mask = num_connections == len(required_indices)
        return self._df[required_target_mask]['id']

    def _load_cards(self) -> pd.DataFrame:
        """
        Loads the DataFrame of main deck monster cards information
        including their ID, name, type, attribute, level, attack, defense, and img_url.
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

    def _calculate_to_adjacency_matrix(self) -> np.ndarray:
        """
        Creates the Small World graph adjacency matrix for all cards.
        Two cards are considered adjacent if they have exactly one property in common
        from the following attributes: type, attribute, level, attack, or defense.

        Returns:
            np.array: An adjacency matrix representing the connections between cards.
        """
        required_columns = ['type', 'attribute', 'level', 'atk', 'def']
        if not all(column in self._df.columns for column in required_columns):
            raise ValueError("DataFrame must have columns: 'type', 'attribute', 'level', 'atk', 'def'")

        # Extract relevant columns and convert to numpy array
        card_attributes = self._df[required_columns].to_numpy()

        # Broadcasting to compare each card with every other card
        # This creates a 3D array where the third dimension is the attribute comparison between cards
        comparisons = card_attributes[:, np.newaxis, :] == card_attributes

        # Sum along the last axis to count the number of similarities between each pair of cards
        similarity_count = comparisons.sum(axis=2)

        # Create the adjacency matrix where exactly one attribute matches
        adjacency_matrix = (similarity_count == 1).astype(int)

        return adjacency_matrix

class Deck:
    """
    Contains data for deck relevant for Small World.
    Card information for all_cards must be provided,
    in addition to either a path to a .ydk (yugioh deck) file or a list of card ids.
    A .ydk file can be downloaded from most deck building websites such as https://ygoprodeck.com/deckbuilder/
    """
    def __init__(self, all_cards: AllCards, ydk_path: Path = None, card_ids: list[int] = None):
        if ydk_path is None and card_ids is None:
            raise ValueError("Either a path to a .ydk file or a list of card ids must be provided.")
        if ydk_path is not None:
            card_ids = utils.ydk_to_card_ids(ydk_path)

        self._df: pd.DataFrame = utils.sub_df(all_cards.get_df(), card_ids, 'id')
        deck_indices = self._df.index
        self._adjacency_matrix: np.ndarray = all_cards.get_adjacency_matrix()[deck_indices,:][:,deck_indices]
        self._squared_adjacency_matrix: np.ndarray = None
        self._graph: Graph = None

    def __len__(self):
        return len(self._df)

    def get_df(self) -> pd.DataFrame:
        """Returns dataframe of deck."""
        return self._df

    # Adjacency Matrix

    def get_adjacency_matrix(self, squared: bool = False) -> np.ndarray:
        """
        Returns the Small World adjacency matrix for deck.
        Returns the square of the Small World adjacency matrix if squared is True.
        Generates it if it has not been calculated yet.
        """
        if not squared:
            return self._adjacency_matrix

        if self._squared_adjacency_matrix is None:
            self._squared_adjacency_matrix = self._adjacency_matrix @ self._adjacency_matrix
        return self._squared_adjacency_matrix

    def get_labeled_adjacency_matrix(self, squared: bool = False) -> pd.DataFrame:
        """Returns adjacency matrix labeled with card names.
        If squared is True, return the squared adjacency matrix."""
        card_names = self._df['name'].to_list()
        adjacency_matrix = self.get_adjacency_matrix(squared)
        return pd.DataFrame(adjacency_matrix, index=card_names, columns=card_names)

    # Graph

    def get_graph(self) -> Graph:
        """Returns the Small World graph for the deck."""
        if self._graph is None:
            self._graph = nx.from_numpy_array(self._adjacency_matrix)
        return self._graph

    def set_card_images(self) -> None:
        """Sets the card image arrays as graph values."""
        if 'image' in self.get_graph().nodes[0]:
            # Images have already been set
            return

        img_urls = self._df['img_url']
        images = utils.load_images(img_urls)
        card_images = utils.normalize_images(images)
        for node_index, card_image in enumerate(card_images):
            self._graph.nodes[node_index]['image'] = card_image

    def get_card_images(self) -> list[np.ndarray]:
        """Returns list of card images corresponding to deck."""
        self.set_card_images()
        card_images = []
        for node_index in self._graph.nodes:
            card_image = self._graph.nodes[node_index]['image']
            card_images.append(card_image)
        return card_images

class Bridges:
    """
    Contains logic for generating bridges for deck.
    If target_ydk_path or target_ids is provided,
    will only consider cards with Small Worlds connections
    to the required targets as valid bridges.
    Otherwise, all cards are considered valid bridges.
    """
    def __init__(self, deck: Deck, all_cards: AllCards, target_ydk_path: Path=None, target_ids: list[int]=None):
        # If ydk path is provided, use it to generate target ids
        if target_ydk_path is not None:
            target_ids = utils.ydk_to_card_ids(target_ydk_path)
        if target_ids is not None:
            bridge_ids = all_cards.filter_required_targets(target_ids)
            card_pool = Deck(all_cards, card_ids=bridge_ids).get_df()
        else:
            card_pool = all_cards.get_df()

        self._card_pool: pd.DataFrame = card_pool
        self._deck: Deck = deck
        self._bridge_matrix: np.ndarray = self._calculate_bridge_matrix(all_cards)
        self._df: pd.DataFrame = None

    def __len__(self):
        return len(self._df)

    def get_df(self, top: int = None) -> pd.DataFrame:
        """Returns dataframe of cards from card pool with bridge scores and number of bridges to deck."""
        if self._df is None:
            bridge_scores = self._calculate_bridge_scores()
            self._assemble_bridges_df(bridge_scores)
        if top is not None:
            return self._df.head(top)
        return self._df

    def _calculate_bridge_matrix(self, all_cards: AllCards) -> np.ndarray:
        """
        Constructs a bridge matrix from a given deck and card pool dataframes.

        The bridge matrix is a subset of the full adjacency matrix of all cards, representing 
        connections between monsters in the deck and those satisfying the required connections.

        If there are m cards in the pool and n cards in the deck, the result should be n x m

        
        Returns:
        - np.ndarray: The bridge matrix indicating connections between deck monsters and bridge monsters.
        """
        # Get indices of monsters that satisfy all required connections
        pool_indices = self._card_pool.index
        deck_indices = self._deck.get_df().index

        # Construct bridge matrix
        return all_cards.get_adjacency_matrix()[deck_indices,:][:,pool_indices]

    def _calculate_bridge_scores(self) -> np.ndarray:
        """
        Calculates bridge scores for a deck of cards. The score is the number of non-zero entries in the squared 
        adjacency matrix (representing deck connections) adjusted by the bridge matrix,
        normalized by the square of (number of cards in the deck + 1).

        Returns:
        - np.ndarray: Array of calculated bridge scores corresponding to each card in the pool
        """
        deck_size = len(self._deck)

        i,j = np.mgrid[:deck_size, :deck_size]
        outer_product_tensor = self._bridge_matrix[i] * self._bridge_matrix[j] # outer product of connection vectors
        squared_adjacency_matrix = self._deck.get_adjacency_matrix(squared=True)
        deck_connection_tensor = outer_product_tensor + squared_adjacency_matrix[:, :, np.newaxis] # A^2 + x(x.T) for all connection vectors x
        deck_connectivity = deck_connection_tensor.astype(bool).sum(axis=(0,1)) #number of non-zero elements in each slice

        bridge_connection_matrix = self._deck.get_adjacency_matrix() @ self._bridge_matrix
        bridge_connectivity = bridge_connection_matrix.astype(bool).sum(axis=0) #num non-zero elements in each row

        # Formula for bridge score derived from block matrix multiplication.
        bridge_score = (deck_connectivity + 2*bridge_connectivity + 1)/((deck_size+1)**2)
        return bridge_score

    def _assemble_bridges_df(self, bridge_score: list[float]):
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
        bridges_df = self._card_pool.copy()
        number_of_connections = self._bridge_matrix.sum(axis=0)
        bridges_df['number_of_connections'] = number_of_connections
        bridges_df['bridge_score'] = bridge_score

        # Filter out cards with 0 connections.
        bridges_df = bridges_df[bridges_df['number_of_connections'] > 0]

        # Reorder columns.
        bridges_df = bridges_df[['bridge_score', 'number_of_connections', 'name', 'type', 'attribute', 'level', 'atk', 'def']]

        # Reorder rows.
        bridges_df = bridges_df.sort_values(by=['bridge_score', 'number_of_connections', 'name'], ascending=[False, False, True]).reset_index(drop=True)
        self._df = bridges_df

if __name__ == "__main__":
    print(AllCards().top_bridges(150))
