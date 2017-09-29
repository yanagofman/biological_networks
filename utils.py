import re
from difflib import SequenceMatcher


cleanup_regex = re.compile('[^a-zA-Z0-9]')


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def clean_up(value):
    return cleanup_regex.sub('', value)
