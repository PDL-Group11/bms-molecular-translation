from noised_image_generator import svg_to_image
from pathlib import Path
import detection_label_generator as dlg
import os
from rdkit.Chem import Draw
from rdkit import Chem
import rdkit
from PIL import Image
import ray
import numpy as np
import lxml.etree as et
import cssutils
import pandas as pd
import logging
from tqdm import tqdm


@ray.remote
def generate_normal_molecule_image(n, index, labels, save_path):
    a = n * index
    b = n * (index + 1)
    for imol in range(a,b):
        inchi = labels['InChI'][imol]
        mol = Chem.MolFromInchi(
            inchi=inchi, 
            sanitize=True, 
            removeHs=True, 
            logLevel=None,
            treatWarningAsError=False)
        svg = dlg._get_svg(mol)
        # Do some SVG manipulation
        svg = et.fromstring(svg.encode('iso-8859-1'))
        img = svg_to_image(svg)
        img = Image.fromarray((255*(1 - img)).astype(np.uint8))
        img.save(os.path.join(str(save_path), str(imol) + '.png'))
    return

#@ray.remote
#def generate_extra_molecule_image_labels(n, index, labels):
#    def preprocess():
#        print(f"{color.BLUE}Creating COCO-style annotations for both sampled datasets [train, val]{color.BLUE}")
#        counts, unique_atoms_per_molecule = dlg.create_unique_ins_labels(self.data,
#                                                                        overwrite=self.overwrite,
#                                                                        base_path=self.base_path)
#    now = time.time()
#    a = n * index
#    b = n * (index + 1)
#    for imol in range(a, b):


if __name__ == '__main__':
    ray.init()
    PROJECT_DIR = Path('.')
    INPUT_DIR = PROJECT_DIR / 'dataset'
    TMP_DIR = PROJECT_DIR / 'tmp'
    TRAIN_DATA_PATH = INPUT_DIR
    EXTRA_IMG_SAVE_PATH = TMP_DIR / 'extra_normal'
    TRAIN_IMG_SAVE_PATH = TMP_DIR / 'train_normal'
    EXTRA_LABELS_PATH = INPUT_DIR / 'extra_approved_InChIs.csv'
    TRAIN_LABELS_PATH = INPUT_DIR / 'train_labels.csv'
    TMP_DIR.mkdir(exist_ok=True)
    EXTRA_IMG_SAVE_PATH.mkdir(exist_ok=True)

    cssutils.log.setLevel(logging.CRITICAL)

    np.set_printoptions(edgeitems=30, linewidth=180)
    print('RDKit version:', rdkit.__version__)
    # Use a specific version of RDKit with known characteristics so that we can reliably manipulate output SVG.
    # assert rdkit.__version__ == '2020.03.6'

    EXTRA_LABELS = pd.read_csv(EXTRA_LABELS_PATH)
    TRAIN_LABELS = pd.read_csv(TRAIN_LABELS_PATH)

    LABELS = TRAIN_LABELS 
    print(f'Read {len(LABELS)} training labels.')
    labels = ray.put(LABELS)
    #results = [generate_normal_molecule_image.remote(67338, i, labels, TMP_DIR / 'extra_normal/') for i in range(36)]
    results = ray.get([generate_normal_molecule_image.remote(67338, i, labels, EXTRA_IMG_SAVE_PATH) for i in range(36)])
    print('DONE!')