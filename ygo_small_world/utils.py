"""Small World utility functions."""
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from PIL import Image
from ygo_small_world.config import SETTINGS

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

    if not pd.Series(column_values).isin(df[column_name]).any():
        raise ValueError("No values are in df. Data may need to be updated.")

    mask = df[column_name].isin(column_values)
    return df.loc[mask].copy()

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

def load_images(urls: list[str]) -> list[np.ndarray]:
    """
    Loads multiple images from a list of URLs.

    Parameters:
        urls (list): A list of URLs of the images.

    Returns:
        list: A list of numpy arrays representing the images.
    """
    images = []
    for url in urls:
        res = requests.get(url, timeout=10)
        image = np.array(Image.open(BytesIO(res.content)))
        images.append(image)
    return images

def normalize_images(images: list[np.ndarray]) -> list[np.ndarray]:
    """
    Normalizes a list of images to a standard size.
    This is mostly relevant for pendulum cards which have a non-standard image size.

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
