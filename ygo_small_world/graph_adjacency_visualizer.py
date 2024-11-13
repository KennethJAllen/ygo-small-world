"""
Module: graph_adjacency_visualizer

This module is a part of the YGO-small-world project, aimed at visualizing the 'Small World' adjacency relationships in Yu-Gi-Oh! decks.
It provides tools for creating and displaying graphs that represent potential 'Small World' bridges and their connections.

Key Functions:
- create_graph: Constructs a graph from deck data.
- display_graph: Renders the graph for visualization.

Usage: Used for graphically representing the Small World connections between Yu-Gi-Oh! cards, aiding in strategic deck building.

Note: Effective for visual analysis of deck structures in relation to Small World card strategies.
Note: Understanding of Yu-Gi-Oh! card properties and Small World mechanics is essential.
"""

import json
from pathlib import Path
from functools import cache
from io import BytesIO
from collections import namedtuple

import requests
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ygo_small_world import small_world_bridge_generator as sw

Settings = namedtuple("Settings", ["card_size", "max_pixel_brightness"])
SETTINGS = Settings(624, 255)

def names_to_image_urls(card_names: list[str]) -> list[str]:
    """
    Retrieves the URLs of the images corresponding to the given card names.

    Parameters:
        card_names (list): A list of card names.

    Returns:
        list: A list of URLs corresponding to the card images.
    """
    current_dir = Path.cwd()
    cardinfo_path = current_dir / "data" / "cardinfo.json"

    # Load the contents of cardinfo.json
    with open(cardinfo_path, 'r', encoding='utf-8') as file_path:
        json_all_cards = json.load(file_path)
    df_all_cards = pd.DataFrame(json_all_cards['data']) #dataframe of all cards to get image links

    df_deck_images = sw.sub_df(df_all_cards, card_names, 'name')
    df_deck_images['card_images'] = df_deck_images['card_images'].apply(lambda x: x[0]['image_url_cropped'])
    urls = df_deck_images['card_images'].tolist()
    return urls

@cache
def load_image(url: str) -> np.ndarray:
    """
    Loads an image from a given URL.

    Parameters:
        url (str): The URL of the image.

    Returns:
        ndarray: A NumPy array representing the image.
    """
    res = requests.get(url, timeout=10)
    imgage = np.array(Image.open(BytesIO(res.content)))
    return imgage

def load_images(urls: list[str]) -> list[np.ndarray]:
    """
    Loads multiple images from a list of URLs.

    Parameters:
        urls (list): A list of URLs of the images.

    Returns:
        list: A list of NumPy arrays representing the images.
    """
    images = []
    for url in urls:
        image = load_image(url)
        images.append(image)
    return images

def normalize_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """
    Normalizes a list of images to a standard size.

    Parameters:
        images (list): A list of NumPy arrays representing the images.

    Returns:
        list: A list of normalized images.
    """
    card_size = SETTINGS.card_size
    max_pixel_brightness = SETTINGS.max_pixel_brightness
    normalized_images = []
    for image in images:
        image_length = image.shape[0]
        image_width = image.shape[1]
        normalized_image = np.ones([card_size, card_size, 3])*max_pixel_brightness
        #covering cases when image is too small
        if image_length < card_size and image_width < card_size: #length & width too small
            normalized_image[:image_length, :image_width, :] = image
        elif image_length < card_size: #only length is too small
            normalized_image[:image_length, :, :] = image[:, :card_size, :]
        elif image_width < card_size: #only width is too small
            normalized_image[:, :image_width, :] = image[:card_size, :, :]
        else: #image is same size or too big
            normalized_image = image[:card_size, :card_size, :]
        normalized_image = normalized_image.astype(np.uint8)
        normalized_images.append(normalized_image)
    return normalized_images

def names_to_images(card_names: list[str]) -> list[np.ndarray]:
    """
    Converts a list of card names to normalized images.

    Parameters:
        card_names (list): A list of card names.

    Returns:
        list: A list of normalized images.
    """
    urls = names_to_image_urls(card_names)
    images = load_images(urls)
    normalized_images = normalize_images(images)
    return normalized_images

def save_image(file_name: str) -> None:
    """
    Saves images to the 'images' folder in the current directory. Assumes a figure

    Parameters:
        file_name (str): The name of the file to save.
    """
    current_dir = Path.cwd()
    images_dir = current_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True) # ensures directory exists
    image_path = images_dir / file_name
    plt.savefig(image_path, dpi=450, bbox_inches='tight')

#### CREATE GRAPH IMAGE ####

def matrix_to_graph(adjacency_matrix: np.ndarray, card_images: list[np.ndarray]) -> nx.Graph:
    """
    Creates a graph from an adjacency matrix and assigns images to nodes.

    Parameters:
        adjacency_matrix (ndarray): A NumPy array representing the adjacency matrix.
        card_images (list): A list of ndarray images for the nodes.

    Returns:
        nx.Graph: A networkx graph with images assigned to nodes.
    """
    if len(card_images) != adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Number of card images must be equal to each dimension of the adjacency matrix.")

    graph = nx.from_numpy_array(adjacency_matrix)
    for node_index, card_image in enumerate(card_images):
        graph.nodes[node_index]['image'] = card_image
    
    return graph

def names_to_graph(card_names: list[str]) -> nx.Graph:
    """
    Converts a list of card names into a graph image visualization and saves it.

    Parameters:
        card_names (list): A list of card names.

    Returns:
        nx.Graph: A networkx graph with images assigned to nodes.
    """
    card_images = names_to_images(card_names)
    df_deck = sw.monster_names_to_df(card_names).reset_index(drop = True)
    adjacency_matrix = sw.df_to_adjacency_matrix(df_deck)
    return matrix_to_graph(adjacency_matrix, card_images)

def ydk_to_graph(ydk_file: str) -> nx.Graph:
    """
    Converts a ydk (Yu-Gi-Oh Deck) file into a graph image visualization and saves it.

    Parameters:
        ydk_file (str): Path to the ydk file of the deck.
    Returns:
        nx.Graph: A networkx graph with images assigned to nodes.
    """
    card_names = sw.ydk_to_monster_names(ydk_file)
    return names_to_graph(card_names)

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
        save_image('graph_image.png')

    plt.show()

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
    if adjacency_max==0:
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

def plot_matrix(full_image: np.ndarray, squared: bool = False, save_image_indicator: bool = False) -> None:
    """
    Plots and saves a matrix image.

    Parameters:
        full_image (ndarray): An image of the adjacency matrix with cards on each axis.
        squared (bool, optional): If True, the image is saved with name referring to the squared adjacency matrix.
    """
    full_image = full_image.astype(np.uint8)

    #create figure
    plt.imshow(full_image)
    ax = plt.subplot(111)
    ax.axis('off')

    if squared & save_image_indicator:
        save_image('small-world-matrix-squared.png')
    elif save_image_indicator:
        save_image('small-world-matrix.png')

    plt.show()
