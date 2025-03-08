"""
This module is aimed at visualizing the 'Small World' adjacency relationships in Yu-Gi-Oh! decks.
It provides tools for creating and displaying graphs that represent potential 'Small World' bridges and their connections.
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pyprojroot import here
from ygo_small_world.small_world_bridge_generator import AllCards, Deck, Bridges
from ygo_small_world import utils
from ygo_small_world.config import SETTINGS

def plot_graph(graph: nx.Graph, save_image_indicator: bool = False) -> None:
    """
    Plots the graph with images as nodes and optionally saves the image.

    Parameters:
        graph (nx.Graph): The networkx graph to be plotted.
        save_image_indicator (bool): Indicator to save the image.
    """
    pos = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect('equal')
    nx.draw_networkx_edges(graph, pos, ax=ax, width=1.3)

    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)

    piesize = max(-0.003 * graph.number_of_nodes() + 0.15, 0.03)
    pie_radius = piesize / 2.0

    trans_data_to_fig = ax.transData.transform
    trans_fig_to_axes = fig.transFigure.inverted().transform

    for node in graph:
        x_fig, y_fig = trans_data_to_fig(pos[node])  # Figure coordinates
        x_ax, y_ax = trans_fig_to_axes((x_fig, y_fig))  # Axes coordinates
        node_ax = plt.axes([x_ax - pie_radius, y_ax - pie_radius, piesize, piesize])
        node_ax.set_aspect('equal')
        node_ax.imshow(graph.nodes[node]['image'])
        node_ax.axis('off')

    ax.axis('off')

    if save_image_indicator:
        fig.save_image('graph_image.png')

    return fig

#### CREATE MATRIX IMAGE ####

def matrix_to_image(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Generate an ndarray of size N x N x 3 that represents an imgae of a matrix.
    N is CARD_SIZE*num_cards. The third dimension is for the color channels.

    Parameters:
        adjacency_matrix (ndarray): A NumPy array representing the adjacency matrix.
    Returns:
        ndarray: An image of the matrix.
    """
    card_size = SETTINGS.card_size
    max_pixel_brightness = SETTINGS.max_pixel_brightness
    num_cards = adjacency_matrix.shape[0]
    #check that the adjacency matrix is square
    if num_cards != adjacency_matrix.shape[1]:
        raise ValueError("The adjacency matrix must be square.")

    matrix_subimage_size = card_size*num_cards #size of matrix subimage, not including card images
    matrix_subimage = np.ones((matrix_subimage_size, matrix_subimage_size, 3))*max_pixel_brightness

    matrix_maximum = np.max(adjacency_matrix)

    # Normalizing the matrix
    normalized_matrix = adjacency_matrix / matrix_maximum

    # Creating a 3D block for each cell
    cell_blocks = max_pixel_brightness * (1 - normalized_matrix[:, :, np.newaxis])
    cell_blocks_repeated = np.repeat(np.repeat(cell_blocks, card_size, axis=0), card_size, axis=1)

    # Creating the final image by repeating these blocks
    matrix_subimage = np.tile(cell_blocks_repeated, (1, 1, 3))
    return matrix_subimage


def cards_and_matrix_to_full_image(adjacency_matrix: np.ndarray, card_names: list[str]) -> np.ndarray:
    """
    Converts an adjacency matrix into an image and saves it.

    Parameters:
        adjacency_matrix (ndarray): A NumPy array representing the ajdacency matrix.
        card_names (list): A list of card names to plot.
    Returns:
        ndarray: An image of the adjacency matrix with cards on each axis.
    """
    card_size = SETTINGS.card_size
    max_pixel_brightness = SETTINGS.max_pixel_brightness
    num_cards = adjacency_matrix.shape[0]

    # Check that number of cards equals each dimension of the adjacency matrix
    if not num_cards == adjacency_matrix.shape[0] == adjacency_matrix.shape[1]:
        raise ValueError("The number of card images must equal to each dimension of the adjacency matrix.")

    # If the adjacency matrix is all zeros, then there are no Small World connections between cards.
    adjacency_max = np.max(adjacency_matrix)
    if adjacency_max == 0:
        raise ValueError("There are no Small World connections between cards.")

    # Create matrix subimage
    matrix_subimage = matrix_to_image(adjacency_matrix)

    # Add card images to axes
    full_image_size = card_size*(num_cards+1)
    full_image = np.ones((full_image_size,full_image_size,3))*max_pixel_brightness

    card_images = names_to_images(card_names)
    vertical_cards = np.concatenate(card_images, axis=1) #concatenated images horizontally
    horizontal_cards = np.concatenate(card_images, axis=0) #concatenated images vertically

    full_image[card_size:, 0:card_size, :] = horizontal_cards
    full_image[0:card_size, card_size:, :] = vertical_cards
    full_image[card_size:, card_size:, :] = matrix_subimage

    return full_image

def names_to_matrix_image(card_names: list[str], squared: bool = False) -> np.ndarray:
    """
    Converts a list of card names into a matrix image.

    Parameters:
        card_names (list): A list of card names to plot.
        squared (bool, optional): If True, the adjacency matrix is squared.
    Returns:
        ndarray: An image of the adjacency matrix with cards on each axis.
    """
    df_deck = sw.monster_names_to_df(card_names).reset_index(drop = True)
    adjacency_matrix = sw.df_to_adjacency_matrix(df_deck, squared=squared)
    # returns image of matrix
    return cards_and_matrix_to_full_image(adjacency_matrix, card_names)

def ydk_to_matrix_image(ydk_file: str, squared: bool = False) -> np.ndarray:
    """
    Converts a ydk file into a matrix image.

    Parameters:
        ydk_file (str): Path to the ydk file of the deck.
        squared (bool, optional): If True, the adjacency matrix is squared.
    Returns:
        ndarray: An image of the adjacency matrix with cards on each axis.
    """
    card_names = sw.ydk_to_monster_names(ydk_file)
    return names_to_matrix_image(card_names, squared=squared)

def plot_matrix(full_image: np.ndarray, squared: bool = False, save_image: bool = False) -> None:
    """
    Plots and saves a matrix image.

    Parameters:
        full_image (ndarray): An image of the adjacency matrix with cards on each axis.
        squared (bool, optional): If True, the image is saved with name referring to the squared adjacency matrix.
    """
    full_image = full_image.astype(np.uint8)

    #create figure
    fig, ax = plt.subplots()
    ax.imshow(full_image)
    ax.axis('off')

    if save_image:
        file_name = 'small-world-matrix-squared.png' if squared else 'small-world-matrix.png'
        images_dir = here() / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_path = images_dir / file_name
        plt.savefig(image_path, dpi=450, bbox_inches='tight')

    return fig
