import numpy as np
import cv2
import os
import pandas as pd
import json
import multiprocessing
from rdkit.Chem import Draw
from rdkit import Chem
from xml.dom import minidom
from collections import defaultdict
from pqdm.processes import pqdm
from scipy.spatial.ckdtree import cKDTree
### for test
from data_loader import MoleculeDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, functional
import matplotlib.pyplot as plt
###


def _get_svg_doc(mol):
    """
    Draws molecule a generates SVG string.
    :param mol:
    :return:
    """
    dm = Draw.PrepareMolForDrawing(mol)
    d2d = Draw.MolDraw2DSVG(300, 300)
    d2d.DrawMolecule(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()

    doc = minidom.parseString(svg)
    return doc

def _get_svg(mol):
    """
    Draws molecule a generates SVG string.
    :param mol:
    :return:
    """
    dm = Draw.PrepareMolForDrawing(mol)
    d2d = Draw.MolDraw2DSVG(300, 300)
    d2d.DrawMolecule(dm)
    d2d.AddMoleculeMetadata(dm)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    return svg 

def _get_unique_atom_inchi_and_rarity(inchi):
    '''
    helper function:
    get the compound unique atom inchi in the format [AtomType + FormalCharge]
    and a dictionary of the metrics taken into account for rarity measures.
    :param inchi: InChI string
    :return: set of atom inchi string
    '''
    mol = Chem.MolFromInchi(inchi)
    assert mol, f'Invalid InChI string: {inchi}'

    doc = _get_svg_doc(inchi)
    # get atom positions in order to oversample hard cases
    atoms_pos = np.array([
        [
        int(round(float(path.getAttribute('drawing-x')), 0)),
        int(round(float(path.getAttribute('drawing-y')), 0))
        ] for path in doc.getElementsByTagName('rdkit:atom')
    ])
    # calculate the minimum distance between atoms ain the molecule
    sampling_weights = {}
    xys = atoms_pos
    kdt = cKDTree(xys)
    dists, neighbors = kdt.query(xys, k=2)
    nearest_dist = dists[:, 1]

    # min distance
    sampling_weights['global_minimum_dist'] = 1 / (np.min(nearest_dist) + 1e-12)
    # number of atoms closer than half of the average distance
    sampling_weights['n_close_atoms'] = np.sum(nearest_dist < np.mean(neareset_dist) * 0.5)
    # average atom degree
    sampling_weights['average_degree'] = np.array([a.GetDegree() for a in mol.GetAtoms()]).mean()
    # number of triple bonds
    sampling_weights['triple_bonds'] = sum([1 for b in mol.GetBonds() if b.GetBondType().name == 'TRIPLE'])
    return [''.join([a.GetSymbol(), str(a.GetFormalCharge())]) for a in mol.GetAtoms()], sampling_weights

def sample_balanced_datasets(data, counts, unique_atoms_per_molecule, datapoints_per_label=2000):
    '''
    Construct a balanced dataset by sampling every label uniformly.
    Returns train and val data in Pandas DataFrame

    :param data: pandas DataFrame with InChI data. 
    :param counts: Count of each label in the dataset in dict.
    :param unique_atoms_per_molecule: pandas DataFrame with unique atom-inchi in each compount in set.
    :param datapoints_per_label: Molecules to sample per label in int.
    :return: balanced train and val dataset in pandas DataFrame.
    '''
    # merge data with the respective set of unique atoms contained
    data = pd.merge(data, unique_atoms_per_molecule, left_on='InChI', right_on='InChI')
    # create DataFrame to save balanced train data
    balanced_train_data = pd.DataFrame(data=None, columns=data.columns)
    balanced_val_data = pd.DataFrame(data=None, columns=data.columns)
    # sample datapoints per unique label type and append to datasets
    print(f'{color.BLUE} Sampling {datapoints_per_label} points per label type {color.END}')
    for k in counts.keys():
        if k == 'N1':
            sampled_train_data = data[
                data.unique_atoms.apply(lambda x: k in x)
            ].sample(5 * datapoints_per_label, replace=True)
        else:
            sampled_train_data = data[
                data.unique_atoms.apply(lambda x: k in x)
            ].sample(datapoints_per_label, replace=True)
        sampled_val_data = data[
            data.unique_atoms.apply(lambda x: k in x)
        ].sample(datapoints_per_label // 100, replace=True)

    balanced_train_data.drop('unique_atoms', axis=1, inplace=True)
    balanced_val_data.drop('unique_atoms', axis=1, inplace=True)
    return balanced_train_data, balanced_val_data

def sample_images(mol_weights, n=10000):
    '''
    Sample compounds depending on complexity.
    :param mol_weights: pandas DataFrame with img_n
    :param n: number of molecules to sample[int]
    :return: Sampled dataset in pandas DataFrame
    '''
    img_names_sampled = pd.DataFrame.sample(mol_weights, n=n, weights=mol, replace=True)
    return img_names_sampled.index.to_list()

def create_unique_ins_labels(data, overwrite=False, base_path='.'):
    '''
    Create a dictionary with the count of each existent atom-inchi in the train dataset
    and a data frame with the atom-inchi in each compound.

    :param data: Pandas dataframe with columns ['file_name', 'InChI'].
    :param overwrite: overwrite existing JSON file at base_path + '/data/unique_atoms_inchi.json'
    :param base_path: base path of the environment.
    :return: Dict of counts[dict] and dataframe of unique atom-inchi per compound.
    '''
    inchi_list = data.InChI.to_list()
    # check if file exists
    output_counts_path = base_path + '/data/unique_atom_inchi_counts.json'
    output_unique_atoms = base_path + '/data/unique_atoms_per_molecule.csv'
    output_mol_rarity = base_path + '/data/mol_rarity_train.csv'
    if all([os.path.exists(p) for p in [output_counts_path, output_unique_atoms]]):
        if overwrite:
            print(f'{color.BLUE}Output files exists, but overwriting. {color.BLUE}')
        else:
            print(
                f'''{color.BOLD}labels JSON {color.END} already exists, skipping process and reading file.\n
                {color.BLUE} Counts file read from: {color.END} {output_counts_path}\n
                {color.BLUE} Unique atoms file read from: {color.END} {output_unique_atoms}\n
                {color.BLUE} Mol rarity file read from: {color.END} {output_counts_path}\n
                {color.BOLD} overwrite: {overwrite} {color.END}
                '''
            )
            return json.load(open(output_counts_path, 'r')), pd.read_csv(output_unique_atoms)
    assert type(inchi_list) == list, 'Input inchi data type must be list'

    n_jobs = multiprocessing.cpu_count() - 1
    # get unique atom-inchi in each compound and count for sampling later
    result = pqdm(
        inchi_list, 
        _get_unique_atom_inchi_and_rarity,
        n_jobs=n_jobs,
        desc='Calculating unique atom-inchi and rarity'
        )
    result, sample_weights = list(map(list, zip(*result)))
    # save counts
    with open(output_counts_path, 'w') as f_out:
        json.dump(counts, f_out)

    # save sample weights
    sample_weights = pd.DataFrame.from_dict(sample_weights)
    sample_weights.insert(0, 'file_name', data.file_name)
    sample_weights.to_csv(output_mol_rarity, index=False)
    # save unique atoms in each molecule to oversample less represented classes later
    unique_atoms_per_molecule = pd.DataFrame({'InChI': inchi_list, 'unique_atoms': [set(r) for r in result]})
    unique_atoms_per_molecule.to_csv(output_unique_atoms, index=False)
    print(
        f'''{color.BLUE} Counts file saved at: {color.END} {output_counts_path}\n
        {color.BLUE} Unique atoms file saved at: {color>END} {output_unique_atoms}
        ''')
    return counts, unique_atoms_per_molecule

def get_bbox(inchi, unique_labels, atom_margin=12, bond_margin=10):
    """
    Get list of dics with atom-InChI and bounding box [x, y, width, height].
    :param InChI: string 
    :param unique_labels: dic with labels and idx for training.
    :param atom_margin: margin for bbox of atoms.
    :param bond_margin: margin for bbox of bonds.

    :return:
    """
    #print('inchi:', inchi)
    # replace unique labels to decide with label class to search
    labels = defaultdict(int)
    for k, v in unique_labels.items():
        labels[k] = v

    mol = Chem.MolFromInchi(
        inchi=inchi, 
        sanitize=True, 
        removeHs=True, 
        logLevel=None,
        treatWarningAsError=False)
    doc = _get_svg_doc(mol)
    #print('doc.toprettyxml():', doc.toprettyxml())

    # Get X and Y from drawing and type is generated
    # from mol Object, concatenating symbol + formal charge
    atoms_data = [{'x':    int(round(float(path.getAttribute('drawing-x')), 0)),
                   'y':    int(round(float(path.getAttribute('drawing-y')), 0)),
                   'type': ''.join([a.GetSymbol(), str(a.GetFormalCharge())])} for path, a in
                  zip(doc.getElementsByTagName('rdkit:atom'), mol.GetAtoms())]
    #print('atoms_data:', atoms_data)

    annotations = []
    # annotating bonds
    for path in doc.getElementsByTagName('rdkit:bond'):
        # Set all '\' or '/' as single bonds
        ins_type = path.getAttribute('bond-smiles')
        if ins_type == '\\' or ins_type == '/':
            ins_type = '-'
        
        # Make bigger margin for bigger bonds (double and triple)
        margin = bond_margin
        if ins_type == '=' or ins_type == '#':
            margin *= 1.5
        
        # creating bbox coordinates as XYWH
        begin_atm_idx = int(path.getAttribute('begin-atom-idx')) - 1
        end_atom_idx = int(path.getAttribute('end-atom-idx')) - 1
        # left-most pos
        x = min(atoms_data[begin_atm_idx]['x'], atoms_data[end_atom_idx]['x']) - margin // 2 
        # up-most pos
        y = min(atoms_data[begin_atm_idx]['y'], atoms_data[end_atom_idx]['y']) - margin // 2 
        width = abs(atoms_data[begin_atm_idx]['x'] - atoms_data[end_atom_idx]['x']) + margin
        height = abs(atoms_data[begin_atm_idx]['y'] - atoms_data[end_atom_idx]['y']) + margin
        annotation = {
            'bbox': [x, y, width, height],
            'category_id': labels[ins_type],
        }
        annotations.append(annotation)
    
    # annotating atoms
    for atom in atoms_data:
        margin = atom_margin
        # better to predict close carbons (2 close instances affected by NMS)
        if atom['type'] == 'C0':
            margin /= 2
        # because of the hydrogens normally the + sign falls out of the box
        if atom['type'] == 'N1':
            margin *= 2
        annotation = {
            'bbox': [
                atom['x'] - margin,
                atom['y'] - margin,
                margin * 2,
                margin * 2
            ],
            'category_id': labels[atom['type']]
        }
        annotations.append(annotation)
        
    return annotations

def plot_bbox(inchi, labels):
    '''
    Plot bounding boxes for inchi in opencv
    :param inchi: InChI string. [str]
    :param labels: Predicted bounding boxes. [dict]
    :return:
    '''
    # create mol image and create np array
    mol = Chem.MolFromInchi(inchi)
    img = np.array(Draw.MolToImage(mol))
    # draw rects
    annotations = get_bbox(inchi, labels)
    for ins in annotations:
        ins_type = ins['category_id']
        x, y, w, h = ins['bbox']
        cv2.rectangle(img, (x, y), (x + w, y + h), np.random.rand(3, ), 2)
    cv2.namedWindow(inchi, cv2.WINDOW_NORMAL)
    cv2.imshow(inchi, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

'''
if __name__ == '__main__':
    train_dataset = MoleculeDataset(
        root='./dataset/train/',
        csv='./dataset/train_labels.csv',
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )
    for i, data in enumerate(train_dataloader):
        print('input image size:', data[0].size())
        print('class label:', data[1])
        print('data[0]:', data[0])
        img = data[0]#[:]
        print('image size: ', img.size())
        print("max: {}, min: {}".format(np.max(img.cpu().numpy()), np.min(img.cpu().numpy())))
        get_bbox(inchi=data[1][0])
        plt.imshow(functional.to_pil_image(img.squeeze(0)))
        plt.show()
        break
'''