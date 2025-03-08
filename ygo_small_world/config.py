"""Small World configuration settings."""
from collections import namedtuple

Settings = namedtuple("Settings", ["card_size", "max_pixel_brightness"])
SETTINGS = Settings(624, 255)
