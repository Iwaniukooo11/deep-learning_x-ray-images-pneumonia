from skimage.io import imread_collection
import pandas as pd

def create_images_labels():
    normal = len(imread_collection('data/test/normal/*.jpeg'))
    pneumonia = len(imread_collection('data/test/pneumonia/*.jpeg'))
    return pd.DataFrame({'label': [0, 1], 'count': [normal, pneumonia]})