import os
import ase
import torch
import random
import pickle

from ase import Atoms
from tqdm import tqdm
from spookynet import SpookyNetCalculator

# # SpookyNet input format
# Z (LongTensor [N]):
#     Nuclear charges (atomic numbers) of atoms.
# Q (FloatTensor [B]):
#     Total charge of each molecule in the batch.
# S (FloatTensor [B]):
#     Total magnetic moment of each molecule in the batch. For
#     example, a singlet has S=0, a doublet S=1, a triplet S=2, etc.
# R (FloatTensor [N, 3]):
#     Cartesian coordinates (x,y,z) of atoms.
# idx_i (LongTensor [P]):
#     Index of atom i for all atomic pairs ij. Each pair must be
#     specified as both ij and ji.
# idx_j (LongTensor [P]):
#     Same as idx_i, but for atom j.

def calculate_atom(_calc, _atoms, _device="cuda"):
    _calc._update_neighborlists(_atoms)
    Z = torch.from_numpy(_atoms.get_atomic_numbers()).long().to(_device)
    Q = torch.Tensor([0]).float().to(_device)
    S = torch.Tensor([0]).float().to(_device)
    R = torch.from_numpy(_atoms.get_positions()).float().to(_device)
    idx_i, idx_j = _calc.idx_i.long().to(_device), calc.idx_j.long().to(_device)

    return _calc.spookynet.energy(Z, Q, S, R, idx_i, idx_j) # energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6

def train(_epoch, _calc, _opt, _data, _num=1000, key="Sum of electronic and thermal Free Energies"):
    _calc.spookynet.train()
    random.shuffle(_data)

    total_loss, total_record = 0, []
    for i in tqdm(range(_num), ncols=80):
        m = _data[i]
        mol = Atoms("".join([num_to_elem[i] for i in m["atoms"]]), positions=m["coord"])
        energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = calculate_atom(_calc, mol, _calc.spookynet.device)

        loss = torch.nn.functional.mse_loss(energy, torch.Tensor([m[key]/27.2113961318]).to(_calc.spookynet.device))
        total_record.append([str(energy.item()), str(m[key]/27.2113961318)])
        _opt.zero_grad()
        loss.backward()
        _opt.step()

        total_loss += loss.item()/_num

    print("The training loss is", total_loss)
    return total_loss, total_record

def test(_epoch, _calc, _data, _num=100, key="Sum of electronic and thermal Free Energies"):
    _calc.spookynet.eval()
    random.shuffle(_data)

    total_loss, total_record = 0, []
    for i in tqdm(range(_num), ncols=80):
        m = _data[i]
        mol = Atoms("".join([num_to_elem[i] for i in m["atoms"]]), positions=m["coord"])
        energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = calculate_atom(_calc, mol, _calc.spookynet.device)
        loss = torch.nn.functional.mse_loss(energy, torch.Tensor([m[key]/27.2113961318]).to(_calc.spookynet.device))
        total_record.append([str(energy.item()), str(m[key]/27.2113961318)])
        total_loss += loss.item()/_num

    print("The testing loss is", total_loss)
    return total_loss, total_record

if __name__ == "__main__":
    """Dictionary of atomic numbers."""
    all_atoms = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    elem_to_num = dict(zip(all_atoms, range(1, len(all_atoms)+1)))
    num_to_elem = dict(zip(range(1, len(all_atoms)+1), all_atoms))

    calc = SpookyNetCalculator(load_from="parameters.pth", charge=0, magmom=0)
    calc.spookynet = torch.load("./weight/20221207_self_trained_5_0030_44.245009671651175_7.840168288052662.pt")
    calc.spookynet.train()

    opt = torch.optim.Adam(calc.spookynet.parameters(), lr=1.5e-3)

    # pkls = os.listdir("/data/lrl/QM_sym_pickles/") * 5
    # pkls = ["/data/lrl/NAS_503_share/BindingDB_pickle/" + i for i in os.listdir("/data/lrl/NAS_503_share/BindingDB_pickle/")] + \
    #        ["/data/lrl/QM_sym_pickles/" + i for i in os.listdir("/data/lrl/QM_sym_pickles/")] + \
    #        ["/data/lrl/NAS_503_share/ChemBL_pickle/" + i for i in os.listdir("/data/lrl/NAS_503_share/ChemBL_pickle/")]
    pkls = ["/data/lrl/NAS_503_share/BindingDB_pickle/" + i for i in os.listdir("/data/lrl/NAS_503_share/BindingDB_pickle/")] + \
           ["/data/lrl/NAS_503_share/ChemBL_pickle/" + i for i in os.listdir("/data/lrl/NAS_503_share/ChemBL_pickle/")]
    pkls = pkls * 4
    random.shuffle(pkls)

    with open("./training_process/20221207_self_trained_6.txt", "w+") as f:
        for i, pkl in enumerate(pkls):
            pkl = pickle.load(open(pkl, "rb"))
            train_loss, train_record = train(_epoch=i, _calc=calc, _opt=opt, _data=pkl, _num=256)
            test_loss, test_record = test(_epoch=i, _calc=calc, _data=pkl, _num=64)
            torch.save(calc.spookynet, "./weight/20221207_self_trained_6_{}_{}_{}.pt".format(str(i).rjust(4, "0"), train_loss, test_loss))

            f.write("EPOCH " + str(i).rjust(4, "0")+"\n")
            f.write("## TRAIN ##"+"\n")
            for r in train_record:
                f.write(str(r[0]) + "\t" + str(r[1]) + "\n")
            f.write("## TEST ##"+"\n")
            for r in test_record:
                f.write(str(r[0]) + "\t" + str(r[1]) + "\n")

            # exit()

    # for pkl in pkls:
    #     pkl = pickle.load(open("/data/lrl/QM_sym_pickles/" + pkl, "rb"))
    #     random.shuffle(pkl)

    #     total_loss, total_num = 0, 0
    #     for m in tqdm(pkl, ncols=80):
    #         mol = Atoms("".join([num_to_elem[i] for i in m["atoms"]]), positions=m["coord"])
    #         energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = calculate_atom(calc, mol, calc.spookynet.device)

    #         loss = torch.nn.functional.mse_loss(energy, torch.Tensor([m["Sum of electronic and thermal Free Energies"]/27.2113961318]).to(calc.spookynet.device))
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #         total_loss += loss.item()
    #         total_num += 1

    #     print("The total loss is", total_loss)