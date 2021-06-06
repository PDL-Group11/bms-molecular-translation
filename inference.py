import copy

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial import cKDTree
from tqdm import tqdm

import cv2
import einops
import torch
import torchvision
import csv
import os
from rdkit.Chem import Draw
from rdkit import RDLogger
import ray

label_dict = {
    1: "C0", 2: "C10", 3: "O0", 4: "Si0", 5: "N0", 6: "Br0", 7: "F0", 8: "S0", 9: "H0", 10: "B0",
    11: "I0", 12: "P0", 13: "O1", 14: "Si-1", 15: "C-1", 16: "C1", 17: "B1", 18: "O-1", 19: "N-1", 20: "N1",
    21: "-", 22: "=", 23: "#"
}

bonds_list = [21, 22, 23]
RDLogger.DisableLog('rdApp.*') 

def bbox_to_graph(output, idx_to_labels, bond_labels):
    # calculate atoms mask (pred classes that are atoms/bonds)
    atoms_mask = np.array([True if ins not in bond_labels else False for ins in output['labels']])
    # print(atoms_mask)
    # get atom list
    atoms_list = [idx_to_labels[a] for a in output['labels'][atoms_mask]]
    # print(atoms_list)
    atoms_list = pd.DataFrame({'atom': atoms_list,
                               'x':    (output['boxes'][atoms_mask, 2] - output['boxes'][atoms_mask, 0])/2 + output['boxes'][atoms_mask, 0],
                               'y':    (output['boxes'][atoms_mask, 3] - output['boxes'][atoms_mask, 1])/2 + output['boxes'][atoms_mask, 1]})

    # in case atoms with sign gets detected two times, keep only the signed one
    for idx, row in atoms_list.iterrows():
        # print(len(row), row)
        # CHECK!
        if row.atom[-1] != '0':
            if row.atom[-2] != '-':
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-1])]
            else:
                overlapping = atoms_list[atoms_list.atom.str.startswith(row.atom[:-2])]

            # cKDTree: provides an index into a set of k-dimensional points which can be used to rapidly look up the nearest neighbors of any point
            kdt = cKDTree(overlapping[['x', 'y']])
            dists, neighbours = kdt.query([row.x, row.y], k=2)
            if dists[1] < 7:
                atoms_list.drop(overlapping.index[neighbours[1]], axis=0, inplace=True)

    bonds_list = []

    # get bonds
    for bbox, bond_type, score in zip(output['boxes'][np.logical_not(atoms_mask)],
                                      output['labels'][np.logical_not(atoms_mask)],
                                      output['scores'][np.logical_not(atoms_mask)]):

        if idx_to_labels[bond_type] == 'SINGLE':
            _margin = 5
        else:
            _margin = 8

        # anchor positions are _margin distances away from the corners of the bbox.
        anchor_positions = (bbox + [_margin, _margin, -_margin, -_margin]).reshape([2, -1])
        oposite_anchor_positions = anchor_positions.copy()
        oposite_anchor_positions[:, 1] = oposite_anchor_positions[:, 1][::-1]

        # Upper left, lower right, lower left, upper right
        # 0 - 1, 2 - 3
        anchor_positions = np.concatenate([anchor_positions, oposite_anchor_positions])

        # get the closest point to every corner
        atoms_pos = atoms_list[['x', 'y']].values
        kdt = cKDTree(atoms_pos)
        dists, neighbours = kdt.query(anchor_positions, k=1)

        # check corner with the smallest total distance to closest atoms
        if np.argmin((dists[0] + dists[1], dists[2] + dists[3])) == 0:
            # visualize setup
            begin_idx, end_idx = neighbours[:2]
        else:
            # visualize setup
            begin_idx, end_idx = neighbours[2:]

        bonds_list.append((begin_idx, end_idx, idx_to_labels[bond_type], score))

    return atoms_list.atom.values.tolist(), bonds_list


def mol_from_graph(atoms, bonds):
    """ construct RDKIT mol object from atoms, bonds and bond types
    atoms: list of atom symbols+fc. ex: ['C0, 'C0', 'O-1', 'N1']
    bonds: list of lists of the born [atom_idx1, atom_idx2, bond_type, score]
    """

    # create and empty molecular graph to add atoms and bonds
    mol = Chem.RWMol()
    nodes_idx = {}
    bond_types = {'-':   Chem.rdchem.BondType.SINGLE,
                  '=':   Chem.rdchem.BondType.DOUBLE,
                  '#':   Chem.rdchem.BondType.TRIPLE}
                #   'AROMATIC': Chem.rdchem.BondType.AROMATIC}

    # add nodes
    for idx, node in enumerate(atoms):
        # neutral formal charge
        if ('0' in node) or ('1' in node):
            a = node[:-1]
            fc = int(node[-1])
        if '-1' in node:
            a = node[:-2]
            fc = -1
        # create atom object
        try:
            a = Chem.Atom(a)
        except:
            continue
        a.SetFormalCharge(fc)

        # add atom to molecular graph (return the idx in object)
        atom_idx = mol.AddAtom(a)
        nodes_idx[idx] = atom_idx

    # add bonds
    existing_bonds = set()
    prev_mol = None
    for idx_1, idx_2, bond_type, score in bonds:
        if (idx_1 in nodes_idx) and (idx_2 in nodes_idx):
            if (idx_1, idx_2) not in existing_bonds and (idx_2, idx_1) not in existing_bonds:
                try:
                    mol.AddBond(nodes_idx[idx_1], nodes_idx[idx_2], bond_types[bond_type])
                except:
                    continue
        existing_bonds.add((idx_1, idx_2))

    mol = mol.GetMol()

    return Chem.MolToInchi(mol)

def infer(model, loader, device):
    model.eval()

    results = {'image_id': [], 'InChI': []}
    df = pd.DataFrame(columns=['image_id', 'InChI'])
    for i, item in enumerate(tqdm(loader)):
        img, filename = item
        img = img.to(device)

        with torch.no_grad():
            out = model(img)
        
            # For drawing bbox
            # img = img[0]
            # img = img.type(torch.uint8)
            # img = img.cpu()
            # boxes = out[0]['boxes'].cpu()
            # # _boxes
            # # boxes = []
            # # for bbox in out:
            # #     tmp = []
            # #     for i in len(bbox):
            # #         tmp.append(int(bbox[i]))
            # #     boxes.append(tmp)
            # # print(boxes)
            # bbox_img = torchvision.utils.draw_bounding_boxes(img, boxes)
            # bbox = einops.rearrange(bbox_img, 'c h w -> h w c')
            # img = einops.rearrange(img, 'c h w -> h w c')

            # bbox = bbox.cpu().numpy()
            # img = img.cpu().numpy()
            # print(filename)
            # cv2.imwrite("./img/bbox.png", bbox * 255)
            # cv2.imwrite("./img/img.png", img * 255)

        result = {}
        for key in out[0].keys():
            values = out[0].get(key).cpu().numpy()
            result[key] = values

        atoms, bonds = bbox_to_graph(result, idx_to_labels=label_dict, bond_labels=bonds_list)
        try:
            inchi = mol_from_graph(atoms, bonds)
        except:
            inchi = '1S/H2O/h1H2'

        # post processing inchi text
        index_trailing_semicolon = inchi.find(';;;')
        inchi = inchi[:index_trailing_semicolon] if index_trailing_semicolon != -1 else inchi


        df = df.append(pd.DataFrame.from_dict({'image_id': [filename[0]], 'InChI': [inchi]}))

    return df