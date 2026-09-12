"""Card databases, deck connections, and Small World bridge rankings."""
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from networkx.classes.graph import Graph
from ygo_small_world import utils
from ygo_small_world.connections import BLOCK_SIZE, connection_blocks, connections
from ygo_small_world.data_paths import readable_card_path
from ygo_small_world.update_data import update_card_data


class AllCards:
    """Contains data for main deck monster cards relevant for Small World"""
    def __init__(self):
        self._df: pd.DataFrame = self._load_cards()
        self._adjacency_matrix: np.ndarray | None = None

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
        if self._adjacency_matrix is None:
            self._adjacency_matrix = self._calculate_to_adjacency_matrix()
        return self._adjacency_matrix

    def get_labeled_adjacency_matrix(self):
        """Returns adjacency matrix labeled with card names."""
        card_names = self._df['name'].tolist()
        return pd.DataFrame(self.get_adjacency_matrix(), index=card_names, columns=card_names)

    def top_bridges(self, num: int = 10, reverse: bool = False) -> pd.DataFrame:
        """Returns the top bridges of all cards.
        Optional arguments: reverse to return bottom bridges, num to specify the number of bridges."""
        total_connections = np.zeros(len(self), dtype=np.int64)
        for _, block in connection_blocks(self._df, self._df):
            total_connections += block.sum(axis=0)
        result = self._df.copy()
        result.insert(2, 'num_connections', total_connections)
        return result.sort_values(by=['num_connections'], ascending=reverse).head(num)

    def filter_required_targets(self, required_target_ids: list[int]) -> pd.Series:
        """Filters and returns cards that connect to all required target names"""
        required_ids = list(dict.fromkeys(required_target_ids))
        supported_ids = set(self._df['id'])
        missing = [card_id for card_id in required_ids if card_id not in supported_ids]
        if missing:
            raise ValueError(f"Unknown or unsupported required target IDs: {missing}")
        targets = self._df[self._df['id'].isin(required_ids)]
        required_target_mask = np.ones(len(self), dtype=bool)
        for _, block in connection_blocks(targets, self._df):
            required_target_mask &= block.all(axis=0)
        return self._df[required_target_mask]['id']

    def _load_cards(self) -> pd.DataFrame:
        """
        Loads the DataFrame of main deck monster cards information
        including their ID, name, type, attribute, level, attack, defense, and img_url.
        """
        cardinfo_path = readable_card_path()

        # Pull card data if it doesn't exist
        if not cardinfo_path.exists():
            print("Card data missing, fetching card data.")
            update_card_data()
            cardinfo_path = readable_card_path()

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
        return connections(self._df, self._df)

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

        self._df = all_cards.get_df().loc[all_cards.get_df()['id'].isin(card_ids)].copy()
        if self._df.empty:
            raise ValueError("Deck contains no supported main-deck monsters. Check the IDs or update card data.")
        self._adjacency_matrix: np.ndarray = connections(self._df, self._df)
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
        if len(self) == 0:
            raise ValueError("Cannot plot an empty deck.")
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
            card_pool = all_cards.get_df().loc[all_cards.get_df()['id'].isin(bridge_ids)].copy()
        else:
            card_pool = all_cards.get_df()

        self._card_pool: pd.DataFrame = card_pool
        self._deck: Deck = deck
        self._bridge_matrix: np.ndarray = self._calculate_bridge_matrix()
        self._df: pd.DataFrame = None

    def __len__(self):
        return len(self.get_df())

    def get_df(self, top: int = None) -> pd.DataFrame:
        """Returns dataframe of cards from card pool with bridge scores and number of bridges to deck."""
        if self._df is None:
            bridge_scores = self._calculate_bridge_scores()
            self._assemble_bridges_df(bridge_scores)
        if top is not None:
            return self._df.head(top)
        return self._df

    def _calculate_bridge_matrix(self) -> np.ndarray:
        """Compute the n×m deck-to-candidate connections directly."""
        return connections(self._deck.get_df(), self._card_pool)

    def _calculate_bridge_scores(self) -> np.ndarray:
        """
        Calculates bridge scores for a deck of cards. The score is the number of non-zero entries in the squared 
        adjacency matrix (representing deck connections) adjusted by the bridge matrix,
        normalized by the square of (number of cards in the deck + 1).

        Returns:
        - np.ndarray: Array of calculated bridge scores corresponding to each card in the pool
        """
        deck_size = len(self._deck)

        reachable = self._deck.get_adjacency_matrix(squared=True).astype(bool)
        scores = np.empty(len(self._card_pool), dtype=float)
        for start in range(0, len(scores), BLOCK_SIZE):
            block = self._bridge_matrix[:, start:start + BLOCK_SIZE]
            connected = block.astype(bool)
            # Nonzero entries of A² + xxᵀ, without integer n×n×pool tensors.
            deck_connectivity = (
                reachable[:, :, None] | (connected[:, None, :] & connected[None, :, :])
            ).sum(axis=(0, 1))
            bridge_connectivity = np.count_nonzero(
                self._deck.get_adjacency_matrix() @ block, axis=0
            )
            diagonal = connected.any(axis=0)
            scores[start:start + block.shape[1]] = (
                deck_connectivity + 2 * bridge_connectivity + diagonal
            ) / (deck_size + 1) ** 2
        return scores

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
