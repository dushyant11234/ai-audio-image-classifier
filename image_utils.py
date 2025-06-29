from PIL import Image
import numpy as np

def preprocess_image(file_path):
    img = Image.open(file_path).resize((128, 128))
    img_array = np.array(img)
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)
