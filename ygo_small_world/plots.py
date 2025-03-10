"""
Visualizes Small World adjacency relationships for Yu-Gi-Oh! decks.
Main functions:
    graph_fig for visualizing the Small World graph
    matrix_fig for visualizing the Small World adjacency matrices
Both take Deck objects as arguments from the bridges module.
"""
#pylint: disable=no-member
from pathlib import Path
import numpy as np
import networkx as nx
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from ygo_small_world.bridges import Deck
from ygo_small_world.config import SETTINGS, Settings


def graph_fig(deck: Deck, save_path: Path=None) -> Figure:
    """
    Plots the Samml World graph of a deck of cards.
    Uses card images as nodes.
    Optionally saves the image.

    Parameters:
        deck (Deck): A deck of cards to be plotted.
        img_filepath (Path): Optional path to save image to.
    """
    # Loads card images
    deck.set_card_images()

    graph = deck.get_graph()
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

    if save_path is not None:
        plt.savefig(save_path, dpi=450, bbox_inches='tight')

    return fig

def matrix_fig(deck: Deck, squared: bool=False, save_path: Path=None) -> Figure:
    """
    Plots and saves a matrix image.

    Parameters:
        full_image (ndarray): An image of the adjacency matrix with cards on each axis.
        squared (bool, optional):
            If False, the adjacency matrix is generated.
            If True, the squared adjacency matrix is generated.
        img_filepath (Path, optional): Optional path to save image to.
    """
    adjacency_matrix = deck.get_adjacency_matrix(squared=squared)
    card_images = deck.get_card_images()
    matrix_img = _create_matrix_img(adjacency_matrix, card_images)

    #create figure
    fig, ax = plt.subplots()
    ax.imshow(matrix_img)
    ax.axis('off')

    if save_path is not None:
        plt.savefig(save_path, dpi=450, bbox_inches='tight')

    return fig

def _create_matrix_img(adjacency_matrix: np.ndarray, card_images: list[np.ndarray], settings: Settings=SETTINGS) -> np.ndarray:
    """
    Converts an adjacency matrix into a uint8 numpy array with cards on axes.

    Parameters:
        adjacency_matrix: A NumPy array representing the ajdacency matrix.
        card_images: A list of card names to plot on the axes corresponding to the adjacency matrix.
    Returns:
        ndarray: An image of the adjacency matrix with cards on each axis of type uint8.
    """
    card_size = settings.card_size
    max_pixel_brightness = settings.max_pixel_brightness
    num_cards = adjacency_matrix.shape[0]

    # Check that number of cards equals each dimension of the adjacency matrix
    if not num_cards == adjacency_matrix.shape[0] == adjacency_matrix.shape[1]:
        raise ValueError("The number of card images must equal to each dimension of the adjacency matrix.")

    # If the adjacency matrix is all zeros, then there are no Small World connections between cards.
    adjacency_max = np.max(adjacency_matrix)
    if adjacency_max == 0:
        raise ValueError("There are no Small World connections between cards.")

    # Create matrix subimage
    matrix_subimage = _create_matrix_subimage(adjacency_matrix)

    # Add card images to axes
    full_image_size = card_size*(num_cards+1)
    full_image = np.full((full_image_size, full_image_size, 3),
                         fill_value=max_pixel_brightness,
                         dtype=np.uint8)

    vertical_cards = np.concatenate(card_images, axis=1) #concatenated images horizontally
    horizontal_cards = np.concatenate(card_images, axis=0) #concatenated images vertically

    full_image[card_size:, 0:card_size, :] = horizontal_cards
    full_image[0:card_size, card_size:, :] = vertical_cards
    full_image[card_size:, card_size:, :] = matrix_subimage

    return full_image

def _create_matrix_subimage(adjacency_matrix: np.ndarray, settings: Settings=SETTINGS) -> np.ndarray:
    """
    Turns the adjacency matrix into greyscale uint8 array
    for placement in full image with cards on axes.
    """
    card_size = settings.card_size
    max_pixel_brightness = settings.max_pixel_brightness

    num_cards = adjacency_matrix.shape[0]
    if num_cards != adjacency_matrix.shape[1]:
        raise ValueError("The adjacency matrix must be square.")

    matrix_maximum = np.max(adjacency_matrix)
    if matrix_maximum == 0:
        raise ValueError("There are no Small World connections between cards.")

    # Normalize to [0..1], invert, scale to max brightness
    normalized_matrix = 1.0 - (adjacency_matrix / matrix_maximum)
    matrix_img = (normalized_matrix * max_pixel_brightness).astype(np.uint8)

    # Create a Pillow image
    pil_img = Image.fromarray(matrix_img, mode='L')  # 'L' = 8-bit grayscale

    # Upscale to the final size using nearest‑neighbor
    new_size = (num_cards * card_size, num_cards * card_size)
    pil_resized = pil_img.resize(new_size, resample=Image.NEAREST)

    # Convert to RGB, then back to a NumPy array
    pil_rgb = pil_resized.convert('RGB')
    matrix_subimage = np.array(pil_rgb, dtype=np.uint8)

    return matrix_subimage
