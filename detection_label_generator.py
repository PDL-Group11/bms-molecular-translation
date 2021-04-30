import numpy as np
from rdkit.Chem import Draw
from xml.dom import minidom
###
from data_loader import MoleculeDataset
from torch.utils.data import DataLoader
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

def get_bbox(inchi, atom_margin=12, bond_margin=10):
    """
    Get list of dics with atom-InChI and bounding box [x, y, width, height].
    :param InChI: string 
    :param unique_labels: dic with labels and idx for training.
    :param atom_margin: margin for bbox of atoms.
    :param bond_margin: margin for bbox of bonds.

    :return:
    """
    mol = Chem.MolFromInchi(
        inchi=inchi, 
        sanitize=True, 
        removeHs=True, 
        logLevel=None,
        treatWarningAsError=False)
    doc = _get_svg_doc(mol)

    # Get X and Y from drawing and type is generated
    # from mol Object, concatenating symbol + formal charge
    atoms_data = [{'x':    int(round(float(path.getAttribute('drawing-x')), 0)),
                   'y':    int(round(float(path.getAttribute('drawing-y')), 0)),
                   'type': ''.join([a.GetSymbol(), str(a.GetFormalCharge())])} for path, a in
                  zip(doc.getElementsByTagName('rdkit:atom'), mol.GetAtoms())]

    annotations = []
    # annotating bonds
    for path in doc.getElementsByTagName('rdkit:bond'):
        print('path:', path)


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
        get_bbox(inchi=data[1])
        plt.imshow(functional.to_pil_image(img.squeeze(0)))
        plt.show()
        break