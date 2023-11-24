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
    atom_feature = one_of_k_encoding(atom.GetSymbol(), ATOM_VOCAB)
    atom_feature += one_of_k_encoding(atom.GetDegree(),[0,1,2,3,4])
    atom_feature += one_of_k_encoding(atom.GetTotalNumHs(),[0,1,2,3,4])
    atom_feature += one_of_k_encoding(atom.GetImplicitValence(),[0,1,2,3,4])
    atom_feature += [atom.GetIsAromatic()]
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

def get_molecular_graph(smi):
    graph = dgl.DGLGraph()
    mol = Chem.MolFromSmiles(smi)

    atoms = mol.GetAtoms()
    num_atoms = len(atoms)
    graph.add_nodes(num_atoms)

    atom_feature_list = [get_atom_features(atom) for atom in atoms]
    atom_feature_list = torch.tensor(atom_feature_list, dtype= torch.float64)

    bonds = mol.GetBonds()
    bond_feature_list= []





# Test
smi = 'C=O'
mol = Chem.MolFromSmiles(smi)

atoms = mol.GetAtoms()
atom_features = []
for atom in atoms:
    atom_feature = get_atom_features(atom)
    atom_features.append(atom_feature)
# print(atom_features)

bond_features = []
bonds = mol.GetBonds()
for bond in bonds:
    bond_feature = get_bond_features(bond)
    bond_features.append(bond_feature)
# print(bond_features)

graph = dgl.DGLGraph()
num_atoms = len(atoms)
graph.add_nodes(num_atoms)
atom_features = torch.tensor(atom_features, dtype=torch.float64)
# bond_features = torch.tensor(bond_features, dtype=torch.float64)
graph.ndata['h'] = atom_features
# graph['e_ij'] = bond_features

print(graph.ndata)
# print(graph.edata)

# # Check
# train_set,valid_set,test_set = get_dataset('SARs')
# train = MyDataset(splitted_set=train_set)
# valid = MyDataset(splitted_set=valid_set)
# test = MyDataset(splitted_set=test_set)

# atom vocab을 직접 따내자!
# atom_vocab = set()
# for smi in train.smi_list:
#     mol = Chem.MolFromSmiles(smi)
#     atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
#     atom_vocab.update(atoms)
# for smi in valid.smi_list:
#     mol = Chem.MolFromSmiles(smi)
#     atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
#     atom_vocab.update(atoms) # 생각보다 몇개 안나옴.. 걍 수작업 하는게 더 좋겠다.

# {'Se', 'Co', 'Na', 'S', 'F', 'As', 'N', 'Cl', 'I', 'K', 'C', 'Br', 'Ca', 'Au', 'B', 'Hg', 'O', 'P'}