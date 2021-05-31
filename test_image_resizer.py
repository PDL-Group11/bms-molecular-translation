from pathlib import Path
import os
from PIL import Image
# import ray
import numpy as np

from tqdm import tqdm

PROJECT_DIR = Path('.')
INPUT_DIR = PROJECT_DIR / 'dataset'
TEST_DATA_PATH = INPUT_DIR / 'test'
SAVE_DATA_PATH = INPUT_DIR / 'new_test'
SIZE = 300

# @ray.remote
def generate_normal_molecule_image(dir_path):
    image_names = os.listdir(TEST_DATA_PATH / dir_path)
    for image in image_names:
        with Image.open(TEST_DATA_PATH/ dir_path / image) as img:
            width = img.width
            height = img.height
            htarget = int(300 * float(height) / width)
            wtarget = int(300 * float(width) / height)
            result = Image.new(img.mode, (300,300), 255)
            if width > height: 
                img = img.resize((300, htarget), Image.BICUBIC)
                result.paste(img, (0, (300-htarget) // 2))
            else:
                img = img.resize((wtarget, 300), Image.BICUBIC)
                result.paste(img, ((300-wtarget) // 2, 0))
            result.save( SAVE_DATA_PATH / dir_path / image)
    return

if __name__ == '__main__':
    # ray.init()

    num_directory = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    for i in num_directory:
        for j in num_directory:
            for k in num_directory:
                dir_path = f"{i}/{j}/{k}"
                (SAVE_DATA_PATH / dir_path).mkdir(parents=True,exist_ok=True)
                generate_normal_molecule_image(dir_path)
    # results = ray.get([generate_normal_molecule_image.remote(10, i, labels, EXTRA_IMG_SAVE_PATH) for i in range(36)])
    print('DONE!')