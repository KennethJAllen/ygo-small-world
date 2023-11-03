import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import requests
from functools import cache
from PIL import Image
from io import BytesIO

def sub_df(df: pd.DataFrame, column_values: list, column_name: str) -> pd.DataFrame:
    """
    Creates a subset of the given DataFrame based on specified values in a particular column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame from which the subset will be extracted.
        column_values (list): A list of values to match against the specified column to filter rows.
        column_name (str): The name of the column in which to look for the specified values.

    Returns:
        pd.DataFrame: A new DataFrame containing only the rows where the specified column contains any of the values in 'column_values'.
    """
    #mask = df[column_name].apply(lambda x: any(value for value in column_values if value == x))
    if column_name not in df.columns:
        raise ValueError(f"'{column_name}' is not a valid column in the DataFrame.")
    
    mask = df[column_name].isin(column_values)
    return df.loc[mask].copy()

@cache
def load_main_monsters() -> pd.DataFrame:
    """
    Loads a DataFrame containing information about all main deck monster cards from a JSON file. 
    The JSON file should contain data for all cards. This function filters for the main monster card categories.

    Returns:
        pd.DataFrame: A DataFrame containing information about all main deck monsters, 
                      including their ID, name, type, attribute, level, attack, and defense.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cardinfo_path = os.path.join(current_dir, "cardinfo.json")

    # Load the contents of cardinfo.json
    with open(cardinfo_path, "r") as file_path:
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

def monster_names_to_df(card_names):
    """
    Converts a list of monster card names into a DataFrame containing details of those monsters.

    Parameters:
        card_names (list): List of monster card names as strings.

    Returns:
        pd.DataFrame: A DataFrame containing the information of the specified monster cards.
    """
    main_monsters = load_main_monsters()
    return sub_df(main_monsters, card_names, 'name')

#### READ YDK FILES ####

def ydk_to_card_ids(ydk_file):
    """
    Extracts card IDs from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file.

    Returns:
        list: A list of card IDs as integers.
    """
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

def df_to_adjacency_matrix(df_cards, squared=False):
    """
    Creates an adjacency matrix based on Small World connections for a given DataFrame of cards.
    Two cards are considered adjacent if they have exactly one property in common from the following attributes: type, attribute, level, attack, or defense.

    Parameters:
        df_cards (pd.DataFrame): DataFrame containing information about the cards.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        np.array: An adjacency matrix representing the connections between cards.
    """
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

def names_to_adjacency_matrix(card_names, squared=False):
    """
    Creates an adjacency matrix based on Small World connections for a list of monster card names.

    Parameters:
        card_names (list): List of monster card names.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        np.array: An adjacency matrix representing the connections between the named cards.
    """
    df_cards = monster_names_to_df(card_names)
    adjacency_matrix = df_to_adjacency_matrix(df_cards, squared=squared)
    return adjacency_matrix

def names_to_labeled_adjacency_matrix(card_names, squared=False):
    """
    Creates a labeled adjacency matrix DataFrame based on Small World connections for a given list of monster names.

    Parameters:
        card_names (list): List of monster names.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        pd.DataFrame: A labeled adjacency matrix with both row and column names corresponding to the monster names.
    """
    df_cards = monster_names_to_df(card_names)
    adjacency_matrix = df_to_adjacency_matrix(df_cards)
    if squared==True:
        adjacency_matrix = np.linalg.matrix_power(adjacency_matrix, 2)
    df_adjacency_matrix = pd.DataFrame(adjacency_matrix, index=card_names, columns=card_names)
    return df_adjacency_matrix

def ydk_to_labeled_adjacency_matrix(ydk_file, squared=False):
    """
    Creates a labeled adjacency matrix DataFrame based on Small World connections from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file containing the deck information.
        squared (bool, optional): If True, the adjacency matrix is squared; default is False.

    Returns:
        pd.DataFrame: A labeled adjacency matrix with both row and column names corresponding to the names of monsters in the ydk file.
    """
    card_names = ydk_to_monster_names(ydk_file)
    df_adjacency_matrix = names_to_labeled_adjacency_matrix(card_names, squared=squared)
    return df_adjacency_matrix

#### BRIDGE FINDING ####

def find_best_bridges(deck_monster_names, required_target_names=[]):
    """
    Identifies the best bridges (monsters) that connect the most cards in the deck via Small World
    and connect to all the required targets.

    Parameters:
        deck_monster_names (list): A list of monster names in the deck.
        required_target_names (list, optional): A list of monster names that must be connected to the bridges via Small World.
          Default is an empty list.

    Returns:
        DataFrame: A Pandas DataFrame containing details of the best bridges including bridge score, number of connections,
          name, type, attribute, level, attack, and defense. If no bridges meet the requirements, prints a message and returns None.
    """
    main_monsters = load_main_monsters()
    sw_adjacency_matrix = df_to_adjacency_matrix(main_monsters) #small world adjacency array of all cards
    deck_monster_names = list(set(deck_monster_names) | set(required_target_names)) #union names so required_target_names is a subset of deck_monster_names
    deck_indices = sub_df(main_monsters, deck_monster_names, 'name').index
    required_indices = sub_df(main_monsters, required_target_names, 'name').index #indices of required targets

    num_required_targets = len(required_target_names) #number of cards required to connect with one bridge

    required_monster_matrix = sw_adjacency_matrix[required_indices,:] #array corresponding to required connection monsters by all monsters
    num_bridges_to_required_cards = required_monster_matrix.sum(axis=0) #number of required connections satisfied by all monsters
    required_bridge_mask = num_bridges_to_required_cards==num_required_targets
    df_bridges = main_monsters[required_bridge_mask].copy() #data frame of monsters connecting all required targets
    required_bridge_indices = df_bridges.index #indices of monsters that satisfy all required connections
    if len(df_bridges)==0:
        print('There are no monsters that bridge all required targets.')
        return 
    #subset of adjacency matrix corresponding to (deck monsters) by (monsters with connections to the required cards)
    bridge_matrix = sw_adjacency_matrix[deck_indices,:][:,required_bridge_indices]

    num_deck_bridges = bridge_matrix.sum(axis=0)
    df_bridges['number_of_connections'] = num_deck_bridges

    #calculate bridge score = num non-zero entries in square of adjacency matrix if bridge was included divided by square of num cards in deck + 1
    adjacency_matrix = names_to_adjacency_matrix(deck_monster_names)
    adjacency_matrix_squared = names_to_adjacency_matrix(deck_monster_names, squared=True)

    num_deck_cards = bridge_matrix.shape[0]
    i,j = np.mgrid[0:num_deck_cards,0:num_deck_cards]
    outer_product_tensor = bridge_matrix[i] * bridge_matrix[j] #outer product of connection vectors
    deck_connection_tensor = outer_product_tensor + adjacency_matrix_squared[:,:,np.newaxis] #A^2 + x(x.T) for all connection vectors x
    deck_connectivity = deck_connection_tensor.astype(bool).astype(int).sum(axis=(0,1)) #number of non-zero elements in each slice

    bridge_connection_matrix = adjacency_matrix @ bridge_matrix
    bridge_connectivity = bridge_connection_matrix.astype(bool).astype(int).sum(axis=0) #num non-zero elements in each row

    bridge_score = (deck_connectivity + 2*bridge_connectivity + 1)/((num_deck_cards+1)**2) #formula derived from block matrix multiplication
    df_bridges['bridge_score'] = bridge_score

    #assemble df
    df_bridges = df_bridges[df_bridges['number_of_connections'] > 0]
    df_bridges = df_bridges[['bridge_score', 'number_of_connections', 'name', 'type', 'attribute', 'level', 'atk', 'def']] #reorder columns
    df_bridges = df_bridges.sort_values(by=['bridge_score','number_of_connections','name'], ascending=[False, False, True]).reset_index(drop=True) #reorder rows
    return df_bridges

def find_best_bridges_from_ydk(ydk_file):
    """
    Identifies the best bridges that connect the most cards in the deck from a given ydk (Yu-Gi-Oh Deck) file.

    Parameters:
        ydk_file (str): Path to the ydk file of the deck.

    Returns:
        DataFrame: A Pandas DataFrame containing details of the best bridges. The same as returned by `find_best_bridges`.
    """
    deck_monster_names = ydk_to_monster_names(ydk_file)
    df_bridges = find_best_bridges(deck_monster_names)
    return df_bridges

#### CARD IMAGES ####

CARD_SIZE = 624
MAX_PIXEL_BRIGHTNESS = 255

def names_to_image_urls(card_names):
    """
    Retrieves the URLs of the images corresponding to the given card names.

    Parameters:
        card_names (list): A list of card names.

    Returns:
        list: A list of URLs corresponding to the card images.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cardinfo_path = os.path.join(current_dir, "cardinfo.json")

    # Load the contents of cardinfo.json
    with open(cardinfo_path, "r") as file_path:
        json_all_cards = json.load(file_path)
    df_all_cards = pd.DataFrame(json_all_cards['data']) #dataframe of all cards to get image links

    df_deck_images = sub_df(df_all_cards,card_names,'name')
    df_deck_images['card_images'] = df_deck_images['card_images'].apply(lambda x: x[0]['image_url_cropped'])
    urls = df_deck_images['card_images'].tolist()
    return urls

@cache
def load_image(url):
    """
    Loads an image from a given URL.

    Parameters:
        url (str): The URL of the image.

    Returns:
        ndarray: A NumPy array representing the image.
    """
    res = requests.get(url)
    imgage = np.array(Image.open(BytesIO(res.content)))
    return imgage

def load_images(urls):
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

def normalize_images(images):
    """
    Normalizes a list of images to a standard size.

    Parameters:
        images (list): A list of NumPy arrays representing the images.

    Returns:
        list: A list of normalized images.
    """
    normalized_images = []
    for image in images:
        image_length = image.shape[0]
        image_width = image.shape[1]
        normalized_image = np.ones([CARD_SIZE,CARD_SIZE,3])*MAX_PIXEL_BRIGHTNESS
        #covering cases when image is too small
        if image_length<CARD_SIZE and image_width<CARD_SIZE: #case when length & width are too small
            normalized_image[:image_length,:image_width,:] = image
        elif image_length<CARD_SIZE: #case when only length is too small
            normalized_image[:image_length,:,:] = image[:,:CARD_SIZE,:]
        elif image_width<CARD_SIZE: #case when only width is too small
            normalized_image[:,:image_width,:] = image[:CARD_SIZE,:,:]
        else: #case when image is same size or too big
            normalized_image = image[:CARD_SIZE,:CARD_SIZE,:]
        normalized_image = normalized_image.astype(np.uint8)
        normalized_images.append(normalized_image)
    return normalized_images

def names_to_images(card_names):
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

#### CREATE GRAPH IMAGE ####

def create_folder(folder_name):
    """
    Creates a folder with the specified name in the current directory if it doesn't exist.

    Parameters:
        folder_name (str): The name of the folder to create.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_images(file_name):
    """
    Saves images to the 'images' folder in the current directory.

    Parameters:
        file_name (str): The name of the file to save.
    """
    folder_name = "images"
    create_folder(folder_name)
    current_dir = os.getcwd()
    image_path = os.path.join(current_dir, folder_name, file_name)
    plt.savefig(image_path, dpi=450, bbox_inches='tight')


def matrix_to_graph_image(connection_matrix, card_images):
    """
    Converts a connection matrix into a graph image visualization and saves it.

    Parameters:
        connection_matrix (ndarray): A NumPy array representing the connection matrix.
        card_images (list): A list of images corresponding to the nodes.
    """
    G = nx.from_numpy_array(connection_matrix)
    for i in range(len(card_images)):
        G.nodes[i]['image'] = card_images[i] #asigns image to each node

    pos=nx.circular_layout(G)

    fig=plt.figure(figsize=(5,5))
    ax=plt.subplot(111)
    ax.set_aspect('equal')
    nx.draw_networkx_edges(G,pos,ax=ax, width=1.3), 

    plt.xlim(-1,1)
    plt.ylim(-1,1)

    trans=ax.transData.transform
    trans2=fig.transFigure.inverted().transform

    num_cards = len(card_images)
    piesize = -0.003*num_cards+0.15 #image size is a linear function of the number of cards
    p2=piesize/2.0
    for n in G:
        xx,yy=trans(pos[n]) #figure coordinates
        xa,ya=trans2((xx,yy)) #axes coordinates
        a = plt.axes([xa-p2,ya-p2, piesize, piesize])
        a.set_aspect('equal')
        a.imshow(G.nodes[n]['image'])
        a.axis('off')
    ax.axis('off')

    create_folder("images")
    save_images('small-wolrd-graph.png')
    plt.show()

def names_to_graph_image(card_names):
    """
    Converts a list of card names into a graph image visualization and saves it.

    Parameters:
        card_names (list): A list of card names.
    """
    card_images = names_to_images(card_names)
    df_deck = monster_names_to_df(card_names).reset_index(drop=True)
    connection_matrix = df_to_adjacency_matrix(df_deck)
    matrix_to_graph_image(connection_matrix, card_images)

def ydk_to_graph_image(ydk_file):
    """
    Converts a ydk (Yu-Gi-Oh Deck) file into a graph image visualization and saves it.

    Parameters:
        ydk_file (str): Path to the ydk file of the deck.
    """
    card_names = ydk_to_monster_names(ydk_file)
    names_to_graph_image(card_names)

#### CREATE MATRIX IMAGE ####

def matrix_to_image(connection_matrix, card_images, squared=False, highlighted_columns=[]):
    """
    Converts a connection matrix into an image and saves it.

    Parameters:
        connection_matrix (ndarray): A NumPy array representing the connection matrix.
        card_images (list): A list of images corresponding to the nodes.
        squared (bool, optional): If True, the connection matrix is squared. Default is False.
        highlighted_columns (list, optional): List of columns to be highlighted. Default is an empty list.
    """
    num_cards = len(card_images)

    vertical_cards = np.concatenate(card_images, axis=1) #concatenated images horizontally
    horizontal_cards = np.concatenate(card_images, axis=0) #concatenated images vertically
    image_size = CARD_SIZE*(num_cards+1)
    matrix_subimage_size = CARD_SIZE*num_cards

    full_image = np.ones((image_size,image_size,3))*MAX_PIXEL_BRIGHTNESS

    #card images
    full_image[CARD_SIZE:,0:CARD_SIZE,:] = horizontal_cards
    full_image[0:CARD_SIZE,CARD_SIZE:,:] = vertical_cards

    matrix_subimage = (np.ones((matrix_subimage_size,matrix_subimage_size,3))*MAX_PIXEL_BRIGHTNESS)

    matrix_maximum = np.max(connection_matrix)
    if highlighted_columns != []:
        highlighted_maximum = np.max(connection_matrix[:,highlighted_columns])

    #color in cells
    for i in range(num_cards):
        for j in range(num_cards):
            matrix_entry = connection_matrix[i,j]
            if matrix_entry>0:
                if j in highlighted_columns:
                    #highlighted cell color
                    red_highlight_max = 255
                    green_highlight_max = 220
                    blue_highlight_max = 220
                    matrix_subimage[i*CARD_SIZE:(i+1)*CARD_SIZE,j*CARD_SIZE:(j+1)*CARD_SIZE,0] = red_highlight_max
                    matrix_subimage[i*CARD_SIZE:(i+1)*CARD_SIZE,j*CARD_SIZE:(j+1)*CARD_SIZE,1] = green_highlight_max*(1-matrix_entry/highlighted_maximum)
                    matrix_subimage[i*CARD_SIZE:(i+1)*CARD_SIZE,j*CARD_SIZE:(j+1)*CARD_SIZE,2] = blue_highlight_max*(1-matrix_entry/highlighted_maximum)
                else:
                    #greyscale cell color
                    cell_brightness_max = 220
                    matrix_subimage[i*CARD_SIZE:(i+1)*CARD_SIZE,j*CARD_SIZE:(j+1)*CARD_SIZE,:] = cell_brightness_max*(1-matrix_entry/matrix_maximum)
                    
    full_image[CARD_SIZE:,CARD_SIZE:,:] = matrix_subimage

    full_image = full_image.astype(np.uint8)

    #create figure
    fig = plt.imshow(full_image)
    ax = plt.subplot(111)
    ax.axis('off')

    create_folder("images")
    if squared==False:
        save_images('small-world-matrix.png')
    else:
        save_images('small-world-matrix-squared.png')
    plt.show()

def names_to_matrix_image(card_names, squared=False):
    """
    Converts a list of card names into a matrix image.

    Parameters:
        card_names (list): A list of card names.
        squared (bool, optional): If True, the connection matrix is squared. Default is False.
    """
    card_images = names_to_images(card_names)
    df_deck = monster_names_to_df(card_names).reset_index(drop=True)
    connection_matrix = df_to_adjacency_matrix(df_deck, squared=squared)
    matrix_to_image(connection_matrix, card_images, squared=squared)

def ydk_to_matrix_image(ydk_file, squared=False):
    """
    Converts a ydk file into a matrix image.

    Parameters:
        ydk_file (str): Path to the ydk file of the deck.
        squared (bool, optional): If True, the connection matrix is squared. Default is False.
    """
    card_names = ydk_to_monster_names(ydk_file)
    names_to_matrix_image(card_names, squared=squared)