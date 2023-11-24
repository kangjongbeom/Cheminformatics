import pandas as pd
import torch
import dgl
import pandas

from rdkit import Chem

from tdc.single_pred import ADME
from tdc.single_pred import HTS

seed = 42

# Dataload
def get_dataset(Name,method='scaffold',data_seed = seed):
    data = None
    if Name == 'SARs':
        data =  HTS(name='SARSCoV2_Vitro_Touret')
    elif Name == 'BBBP':
        data = ADME(name='BBB_Martins')
    else:
        return print('Check your Name input, there are just two Datasets (SARs, BBBP)')

    split = data.get_split(method=method,seed=data_seed)

    train_set = split['train']
    valid_set = split['valid']
    test_set = split['test']

    return train_set, valid_set, test_set

# split smi / label
def get_smi_label(dataset):
    smi_list = list(dataset.Drug)
    label_list = list(dataset.Y)
    return smi_list, label_list

# Make Dataset for train
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, splitted_set):
        self.smi_list = list(splitted_set.Drug)
        self.label_list = list(splitted_set.Y)

    def __len__(self): # len() possible
        return len(self.smi_list)

    def __getitem__(self, idx): # indexing possible
        return self.smi_list[idx], self.label_list[idx]


## Featarue Engineering
ATOM_VOCAB = [
	'C', 'N', 'O', 'S', 'F',
	'H', 'Si', 'P', 'Cl', 'Br',
	'Li', 'Na', 'K', 'Mg', 'Ca',
	'Fe', 'As', 'Al', 'I', 'B',
	'V', 'Tl', 'Sb', 'Sn', 'Ag',
	'Pd', 'Co', 'Se', 'Ti', 'Zn',
	'Ge', 'Cu', 'Au', 'Ni', 'Cd',
	'Mn', 'Cr', 'Pt', 'Hg', 'Pb',
    'NA'
]


def one_of_k_encoding(x, vocab):
	if x not in vocab:
		x = vocab[-1] #
	return list(map(lambda s: float(x==s), vocab))

def get_atom_features(atom):
    atom_feature = one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB) # 41
    atom_feature += one_of_k_encoding(atom.GetDegree(),[0,1,2,3,4,5]) # 6
    atom_feature += one_of_k_encoding(atom.GetTotalNumHs(),[0,1,2,3,4]) # 5
    atom_feature += one_of_k_encoding(atom.GetImplicitValence(),[0,1,2,3,4,5]) # 6
    atom_feature += [atom.GetIsAromatic()] #1
    return atom_feature

def get_bond_features(bond):
    bt = bond.GetBondType()
    bond_features = [
        bt == Chem.BondType.SINGLE,
        bt == Chem.BondType.DOUBLE,
        bt == Chem.BondType.TRIPLE,
        bt == Chem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return bond_features

def get_molecular_graph(smi): # graph features
    graph = dgl.DGLGraph()
    mol = Chem.MolFromSmiles(smi)

    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    graph.add_nodes(num_atoms)

    atom_feature_list = [get_atom_features(atom) for atom in atoms]
    atom_feature_list = torch.tensor(atom_feature_list, dtype= torch.float64)
    graph.ndata['h'] = atom_feature_list

    bonds = mol.GetBonds()
    bond_feature_list= []
    for bond in bonds:
        bond_feature = get_bond_features(bond)

        src = bond.GetBeginAtom().GetIdx()
        dst = bond.GetEndAtom().GetIdx()

        # for directional
        graph.add_edges(src, dst)
        bond_feature_list.append(bond_feature)

        graph.add_edges(dst, src)
        bond_feature_list.append(bond_feature)

    bond_feature_list = torch.tensor(bond_feature_list, dtype=torch.float64)
    graph.edata['e_ij'] = bond_feature_list
    return graph

def my_collate_fn(batch): # for dataloader
    graph_list = []
    label_list = []
    for item in batch:
        smi = item[0]
        label = item[1]
        graph = get_molecular_graph(smi)
        graph_list.append(graph)
        label_list.append(label)
    graph_list = dgl.batch(graph_list)
    label_list = torch.tensor(label_list, dtype=torch.float64)
    return graph_list, label_list


def debugging():
    data = HTS(name='SARSCoV2_Vitro_Touret')

    split = data.get_split(
        method='scaffold',
        seed=seed
    )

    train_set = split['train']
    valid_set = split['valid']
    test_set = split['test']

    smi, label = get_smi_label(train_set)
    graph = get_molecular_graph(smi[0])

if __name__ ==  '__main__':
    debugging()