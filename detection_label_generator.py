import numpy as np
import cv2
import os
import pandas as pd
import json
import multiprocessing
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import MolDrawOptions
from xml.dom import minidom
from collections import defaultdict, Counter
from pqdm.processes import pqdm
from scipy.spatial.ckdtree import cKDTree
import skimage.measure
from util import *

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

    doc = _get_svg_doc(mol)
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
    sampling_weights['n_close_atoms'] = np.sum(nearest_dist < np.mean(nearest_dist) * 0.5)
    # average atom degree
    sampling_weights['average_degree'] = np.array([a.GetDegree() for a in mol.GetAtoms()]).mean()
    # number of triple bonds
    sampling_weights['triple_bonds'] = sum([1 for b in mol.GetBonds() if b.GetBondType().name == 'TRIPLE'])
    return [''.join([a.GetSymbol(), str(a.GetFormalCharge())]) for a in mol.GetAtoms()], sampling_weights

def get_mol_sample_weight(data, data_mode='train', p=1000, base_path='.'):
    """
    Creating sampling weights to oversample hard cases based on bond, atoms, overlaps and rings.
    :param data: DataFrame with train data(InChI). [Pandas DF]
    :param data_mode: Train or extra. [str]
    :param p: Rarity weight. [int]
    :param base_path: base path of the environment. [str]
    :return:
    """
    # load rarity file
    mol_rarity_path = base_path + f'/dataset/{data_mode}_mol_rarity.csv'
    assert os.path.exists(mol_rarity_path), 'No mol_rarity.csv. Create it first'
    mol_rarity = pd.read_csv(mol_rarity_path)

    # filter by given list, calculate normalized weight value per image
    mol_rarity = pd.merge(mol_rarity, data, left_on='image_id', right_on='image_id')
    mol_rarity.drop(['InChI'], axis=1, inplace=True)
    mol_rarity.set_index('image_id', inplace=True)

    # sort each column, after filtering, then assign weight values
    for column in mol_rarity.columns:
        mol_rarity_col = mol_rarity[column].values.astype(np.float64)
        mol_rarity_col_sort_idx = np.argsort(mol_rarity_col)
        ranking_values = np.linspace(1.0 / len(mol_rarity_col), 1.0, num=len(mol_rarity_col))
        ranking_values = ranking_values ** p
        mol_rarity_col[mol_rarity_col_sort_idx] = ranking_values
        mol_rarity[column] = mol_rarity_col
    # normalized weights per img
    mol_weights = pd.DataFrame.sum(mol_rarity, axis=1)
    mol_weights /= pd.DataFrame.sum(mol_weights, axis=0) + 1e-12
    return mol_weights

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
    img_names_sampled = pd.DataFrame.sample(mol_weights, n=n, weights=mol_weights, replace=True)
    return img_names_sampled.index.to_list()

def create_unique_ins_labels(data, mode, overwrite=False, base_path='.'):
    '''
    Create a dictionary with the count of each existent atom-inchi in the train dataset
    and a data frame with the atom-inchi in each compound.

    :param data: Pandas dataframe with columns ['file_name', 'InChI'].
    :param overwrite: overwrite existing JSON file at base_path + '/data/unique_atoms_inchi.json'
    :param base_path: base path of the environment.
    :return: Dict of counts[dict] and dataframe of unique atom-inchi per compound.
    '''
    inchi_list = data.InChI.to_list()
    print('')
    # check if file exists
    output_counts_path = base_path + '/dataset/{}_unique_atom_inchi_counts.json'.format(mode)
    output_unique_atoms = base_path + '/dataset/{}_unique_atoms_per_molecule.csv'.format(mode)
    output_mol_rarity = base_path + '/dataset/{}_mol_rarity.csv'.format(mode)
    print('test cp')
    if all([os.path.exists(p) for p in [output_counts_path, output_unique_atoms]]):
        if overwrite:
            print(f'{color.BLUE}Output files exists, but overwriting. {color.BLUE}')
        else:
            print(
                f'''{color.BOLD}labels JSON {color.END} already exists, skipping process and reading file.\n
                {color.BLUE} Counts file read from: {color.END} {output_counts_path}\n
                {color.BLUE} Unique atoms file read from: {color.END} {output_unique_atoms}\n
                {color.BLUE} Mol rarity file read from: {color.END} {output_mol_rarity}\n
                {color.BOLD} overwrite: {overwrite} {color.END}
                '''
            )
            return json.load(open(output_counts_path, 'r')), pd.read_csv(output_unique_atoms)
    assert type(inchi_list) == list, 'Input inchi data type must be list'

    n_jobs = multiprocessing.cpu_count() - 1
    # get unique atom-inchi in each compound and count for sampling later
    #result = _get_unique_atom_inchi_and_rarity(inchi_list[0])
    result = pqdm(
        inchi_list, 
        _get_unique_atom_inchi_and_rarity,
        n_jobs=n_jobs,
        desc='Calculating unique atom-inchi and rarity'
        )
    #result, sample_weights = list(map(list, zip(*result)))
    #print('listmap:', list(map(list, zip(result))))
    result, sample_weights = list(map(list, zip(*result)))
    counts = Counter(x for xs in result for x in xs)
    # save counts
    with open(output_counts_path, 'w') as f_out:
        json.dump(counts, f_out)

    # save sample weights
    sample_weights = pd.DataFrame.from_dict(sample_weights)
    sample_weights.insert(0, 'image_id', data.image_id)
    sample_weights.to_csv(output_mol_rarity, index=False)
    # save unique atoms in each molecule to oversample less represented classes later
    unique_atoms_per_molecule = pd.DataFrame({'InChI': inchi_list, 'unique_atoms': [set(r) for r in result]})
    unique_atoms_per_molecule.to_csv(output_unique_atoms, index=False)
    print(
        f'''{color.BLUE} Counts file saved at: {color.END} {output_counts_path}\n
        {color.BLUE} Unique atoms file saved at: {color.END} {output_unique_atoms}
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
        #print('begin_atm_idx:', begin_atm_idx)
        #print('end_atm_idx:', end_atom_idx)
        #print('len(atoms_data):', len(atoms_data))
        if begin_atm_idx >= len(atoms_data):
            begin_atm_idx = len(atoms_data) - 1
        if end_atom_idx >= len(atoms_data):
            end_atom_idx = len(atoms_data) - 1
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
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
    cv2.namedWindow(inchi, cv2.WINDOW_NORMAL)
    cv2.imshow(inchi, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def preprocess_train_dataset(
    overwrite=False, 
    min_points_threshold=500, 
    n_sample_per_label=20000, 
    n_sample_hard=200000, 
    n_jobs=multiprocessing.cpu_count()-1):

    if not all([os.path.exists(f'./dataset/train_annotations_{mode}.pkl') for mode in ['train', 'val']]):
        print(f"{color.BLUE}Creating COCO-style annotations for both sampled datasets [train, val]{color.BLUE}")
        data_frame = pd.read_csv('./dataset/train_labels.csv')
        data_frame = data_frame

        # Get counts and unique atoms per molecules to construct datasets.
        counts, unique_atoms_per_molecule = create_unique_ins_labels(data_frame, 
                                                                    mode='train',
                                                                    overwrite=overwrite)

        # bonds SMARTS
        unique_bonds = ['-', '=', '#']

        # Choose labels depending on a minimum count.
        counts = {k: v for k, v in counts.items() if v > min_points_threshold}
        labels = list(counts.keys()) + unique_bonds
        unique_labels = {u: idx + 1 for idx, u in zip(range(len(labels)), labels)}

        # Sample uniform datasets among labels
        train_balanced, val_balanced = sample_balanced_datasets(data_frame,
                                                                counts,
                                                                unique_atoms_per_molecule,
                                                                datapoints_per_label=n_sample_per_label)

        # sample hard cases
        sampled_train = sample_images(get_mol_sample_weight(data=data_frame, data_mode='train'), n=n_sample_hard)
        sampled_val = sample_images(get_mol_sample_weight(data=data_frame, data_mode='train'), n=n_sample_hard // 100)

        # create splits with sampled data
        data_frame.set_index('image_id', inplace=True)
        data_train = data_frame.loc[sampled_train].reset_index()
        data_val = data_frame.loc[sampled_val].reset_index()
        #data_train = data_frame.sample(frac=0.8, random_state=1)
        #data_val = data_frame.drop(data_train.index)

        # concatenate both datasets
        data_train = pd.concat([data_train, train_balanced]).drop_duplicates()
        data_train.sort_values(by=['image_id'], inplace=True)
        data_val = pd.concat([data_val, val_balanced]).drop_duplicates()
        data_val.sort_values(by=['image_id'], inplace=True)
        print('len(data_train):', len(data_train))
        print('len(data_val):', len(data_val))

        # create COCO annotations
        for data_split, mode in zip([data_train, data_val], ['train', 'val']):
            if os.path.exists(f'./dataset/train_annotations_{mode}.pkl'):
                f"{color.BLUE}{mode.capitalize()} already exists, skipping...{color.END}"
                continue
            params = [[row.InChI,
                        row.image_id,
                        mode,
                        unique_labels] for _, row in data_split.iterrows()]
            result = pqdm(params,
                            create_train_COCO_json,
                            n_jobs=n_jobs,
                            argument_type='args',
                            desc=f'{color.BLUE}Creating COCO-style {mode} annotations{color.END}')
            #result = create_COCO_json(data_split.iloc[0].InChI, data_split.iloc[0].image_id, 'train', unique_labels)

            # clean any corrupted annotation
            result = [annotation for annotation in result if type(annotation) == dict]
            print(f'{color.PURPLE}Saving COCO annotations - {mode}{color.END}')
            with open(f'./dataset/train_annotations_{mode}.pkl', 'wb') as fout:
                pickle.dump(result, fout)

        print(f'{color.BLUE}Saving training labels{color.END}')
        with open(f'./dataset/train_labels.json', 'w') as fout:
            json.dump(unique_labels, fout)

        return
    else:
        print(f"{color.BLUE}Preprocessed files already exist. Loading annotations... [train, val]{color.END}")
        return

def preprocess_extra_dataset(
    overwrite=False, 
    min_points_threshold=500, 
    n_sample_per_label=20000, 
    n_sample_hard=200000, 
    n_jobs=multiprocessing.cpu_count()-1):

    if not all([os.path.exists(f'./dataset/extra_annotations_{mode}.pkl') for mode in ['train', 'val']]):
        print(f"{color.BLUE}Creating COCO-style extra annotations for both sampled datasets [train, val]{color.BLUE}")
        data_frame = pd.read_csv('./dataset/extra_approved_InChIs.csv')
        data_frame = data_frame
        if 'image_id' not in data_frame:
            data_frame['image_id'] = data_frame.index
        print('len(data_frame):', len(data_frame))

        # Get counts and unique atoms per molecules to construct datasets.
        counts, unique_atoms_per_molecule = create_unique_ins_labels(data_frame, 
                                                                    mode='extra',
                                                                    overwrite=overwrite)

        # bonds SMARTS
        unique_bonds = ['-', '=', '#']

        # Choose labels depending on a minimum count.
        counts = {k: v for k, v in counts.items() if v > min_points_threshold}
        labels = list(counts.keys()) + unique_bonds
        unique_labels = {u: idx + 1 for idx, u in zip(range(len(labels)), labels)}

        # Sample uniform datasets among labels
        train_balanced, val_balanced = sample_balanced_datasets(data_frame,
                                                                counts,
                                                                unique_atoms_per_molecule,
                                                                datapoints_per_label=n_sample_per_label)

        # sample hard cases
        sampled_train = sample_images(get_mol_sample_weight(data=data_frame, data_mode='extra'), n=n_sample_hard)
        sampled_val = sample_images(get_mol_sample_weight(data=data_frame, data_mode='extra'), n=n_sample_hard // 100)

        # create splits with sampled data
        data_frame.set_index('image_id', inplace=True)
        data_train = data_frame.loc[sampled_train].reset_index()
        data_val = data_frame.loc[sampled_val].reset_index()

        # concatenate both datasets
        data_train = pd.concat([data_train, train_balanced]).drop_duplicates()
        data_train.sort_values(by=['image_id'], inplace=True)
        data_val = pd.concat([data_val, val_balanced]).drop_duplicates()
        data_val.sort_values(by=['image_id'], inplace=True)

        # create COCO annotations
        for data_split, mode in zip([data_train, data_val], ['train', 'val']):
            if os.path.exists(f'./dataset/extra_annotations_{mode}.pkl'):
                f"{color.BLUE}{mode.capitalize()} already exists, skipping...{color.END}"
                continue
            params = [[row.InChI,
                        row.image_id,
                        mode,
                        unique_labels] for _, row in data_split.iterrows()]
            result = pqdm(params,
                            create_extra_COCO_json,
                            n_jobs=n_jobs,
                            argument_type='args',
                            desc=f'{color.BLUE}Creating COCO-style extra {mode} annotations{color.END}')
            #result = create_extra_COCO_json(data_split.iloc[0].InChI, data_split.iloc[0].image_id, 'train', unique_labels)

            # clean any corrupted annotation
            result = [annotation for annotation in result if type(annotation) == dict]
            print(f'{color.PURPLE}Saving extra COCO annotations - {mode}{color.END}')
            with open(f'./dataset/extra_annotations_{mode}.pkl', 'wb') as fout:
                pickle.dump(result, fout)

        print(f'{color.BLUE}Saving extra training labels{color.END}')
        with open(f'./dataset/extra_labels.json', 'w') as fout:
            json.dump(unique_labels, fout)

        return
    else:
        print(f"{color.BLUE}Preprocessed files already exist. Loading annotations... [train, val]{color.END}")
        return

def create_train_COCO_json(inchi, image_id, mode, labels, base_path='.'):
    """
    Create COCO style dataset. If there is not image for the smile
    it creates it.
    :param labels:
    :param inchi: InChI. [str]
    :param image_id: Name of the image file. [str]
    :param mode: train or val. [str]
    :param labels: dic with labels and idx for training.
    :param base_path: base path of the environment. [str]
    :return:
    """
    #if not os.path.exists(base_path + f'/dataset/train_detection_{mode}/{file_name}'):
    mol = Chem.MolFromInchi(inchi)
    # Chem.Draw.MolToImageFile(
    #     mol, os.path.join(f'{base_path}/dataset/train_detection/{mode}/', f'{image_id}.png')
    # )
    #plot_bbox(inchi, labels)

    options = MolDrawOptions()
    options.useBWAtomPalette()

    img = Chem.Draw.MolToImage(mol, size=(600,600), options=options)
    fn = lambda x : 0 if x < 100 else 255
    img = img.convert('L').point(fn, mode='1')
    img = np.asarray(img)

    img = skimage.measure.block_reduce(img, (2,2), np.min)
    
    # add white points
    salt_amount = np.random.uniform(0, 10 / mol.GetNumAtoms())
    salt = np.random.uniform(0, 1, img.shape) < salt_amount
    img = np.logical_or(img, salt)

    # add black points
    pepper_amount = np.random.uniform(0, 0.001)
    pepper = np.random.uniform(0, 1, img.shape) < pepper_amount
    img = np.logical_or(1 - img, pepper)
    
    img = img.astype(np.uint8)
    img = Image.fromarray((255*(1 - img)).astype(np.uint8))
    img.save(f'{base_path}/dataset/train_detection/{mode}/{image_id}.png')

    return {'file_name':   f'{base_path}/dataset/train_detection/{mode}/{image_id}.png',
            'height':      300,
            'width':       300,
            'image_id':    image_id,
            'annotations': get_bbox(inchi, labels)}

def create_extra_COCO_json(inchi, image_id, mode, labels, base_path='.'):
    """
    Create COCO style dataset. If there is not image for the smile
    it creates it.
    :param labels:
    :param inchi: InChI. [str]
    :param image_id: Name of the image file. [str]
    :param mode: train or val. [str]
    :param labels: dic with labels and idx for training.
    :param base_path: base path of the environment. [str]
    :return:
    """
    #if not os.path.exists(base_path + f'/dataset/train_detection_{mode}/{file_name}'):
    mol = Chem.MolFromInchi(inchi)
    Chem.Draw.MolToImageFile(
        mol, os.path.join(f'{base_path}/dataset/extra_detection/{mode}/', f'{image_id}.png')
    )
    #plot_bbox(inchi, labels)
    #print('labels:', labels)
    #print('get_bbox(inchi, labels):', get_bbox(inchi, labels))

    return {'file_name':   f'{base_path}/dataset/extra_detection/{mode}/{image_id}.png',
            'height':      300,
            'width':       300,
            'image_id':    image_id,
            'annotations': get_bbox(inchi, labels)}

if __name__ == '__main__':
    train_detection_path = './dataset/train_detection/'
    extra_detection_path = './dataset/extra_detection/'
    if not os.path.exists(train_detection_path):
        os.mkdir(train_detection_path)
        train_detection_train_path = './dataset/train_detection/train/'
        train_detection_val_path = './dataset/train_detection/val/'
        if not os.path.exists(train_detection_train_path):
            os.mkdir(train_detection_train_path)
        if not os.path.exists(train_detection_val_path):
            os.mkdir(train_detection_val_path)
    if not os.path.exists(extra_detection_path):
        os.mkdir(extra_detection_path)
        extra_detection_train_path = './dataset/extra_detection/train/'
        extra_detection_val_path = './dataset/extra_detection/val/'
        if not os.path.exists(extra_detection_train_path):
            os.mkdir(extra_detection_train_path)
        if not os.path.exists(extra_detection_val_path):
            os.mkdir(extra_detection_val_path)
    preprocess_train_dataset()
    # preprocess_extra_dataset()