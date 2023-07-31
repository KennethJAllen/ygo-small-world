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

def sub_df(df, column_values, column_name):
    #creates subset of dataframe consisting of rows with column_values in column
    df = df.copy()
    mask = df[column_name].apply(lambda x: any(value for value in column_values if value == x))
    return df[mask]

def load_main_monsters():
    #loads dataframe of all main deck monsters
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

MAIN_MONSTERS = load_main_monsters()

def monster_names_to_df(card_names):
    #converts list of monster names into a dataframe of those monsters
    df_cards = sub_df(MAIN_MONSTERS, card_names, 'name')
    return df_cards

#### READ YDK FILES ####

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

#### ADJACENCY MATRIX GENERATION ####

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

def names_to_adjacency_matrix(card_names, squared=False):
    df_cards = monster_names_to_df(card_names)
    adjacency_matrix = df_to_adjacency_matrix(df_cards, squared=squared)
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

    #calculate bridge score = num non-zero entries in square of adjacency matrix if bridge was included ...
    # ... divided by square of num cards in deck + 1
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
    #inputs: ydk file of deck
    #output: The bridges that connect the most cards in your deck
    deck_monster_names = ydk_to_monster_names(ydk_file)
    df_bridges = find_best_bridges(deck_monster_names)
    return df_bridges

#### GET CARD IMAGES ####

def names_to_image_urls(card_names):
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
    res = requests.get(url)
    imgage = np.array(Image.open(BytesIO(res.content)))
    return imgage

def load_images(urls):
    images = []
    for url in urls:
        image = load_image(url)
        images.append(image)
    return images

CARD_SIZE = 624
MAX_PIXEL_BRIGHTNESS = 255

def normalize_images(images):
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
    urls = names_to_image_urls(card_names)
    images = load_images(urls)
    normalized_images = normalize_images(images)
    return normalized_images

#### CREATE GRAPH IMAGE ####

def matrix_to_graph_image(connection_matrix, card_images):
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
    plt.savefig('images\small-wolrd-graph.png', dpi=450)
    plt.show()

def names_to_graph_image(card_names):
    card_images = names_to_images(card_names)
    df_deck = monster_names_to_df(card_names).reset_index(drop=True)
    connection_matrix = df_to_adjacency_matrix(df_deck)
    matrix_to_graph_image(connection_matrix, card_images)

def ydk_to_graph_image(ydk_file):
    card_names = ydk_to_monster_names(ydk_file)
    names_to_graph_image(card_names)

#### CREATE MATRIX IMAGE ####

def matrix_to_image(connection_matrix, card_images, squared=False, highlighted_columns=[]):
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
    if squared==False:
        plt.savefig('images\small-world-matrix.png', dpi=450, bbox_inches='tight')
    else:
        plt.savefig('images\small-world-matrix-squared.png', dpi=450, bbox_inches='tight')
    plt.show()

def names_to_matrix_image(card_names, squared=False):
    card_images = names_to_images(card_names)
    df_deck = monster_names_to_df(card_names).reset_index(drop=True)
    connection_matrix = df_to_adjacency_matrix(df_deck, squared=squared)
    matrix_to_image(connection_matrix, card_images, squared=squared)

def ydk_to_matrix_image(ydk_file, squared=False):
    card_names = ydk_to_monster_names(ydk_file)
    names_to_matrix_image(card_names, squared=squared)