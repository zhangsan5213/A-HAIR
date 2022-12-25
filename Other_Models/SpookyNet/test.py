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

def test(_epoch, _calc, _data, _num=100, key="Sum of electronic and thermal Free Energies"):
    _calc.spookynet.eval()

    total_loss, total_record = 0, []
    for i in range(_num):
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

    """ Calculates the loss. """
    calc = SpookyNetCalculator(load_from="parameters.pth", charge=0, magmom=0)

    # pkls = ["/data/lrl/NAS_503_share/BindingDB_pickle/" + i for i in os.listdir("/data/lrl/NAS_503_share/BindingDB_pickle/")] + \
    #        ["/data/lrl/NAS_503_share/ChemBL_pickle/" + i for i in os.listdir("/data/lrl/NAS_503_share/ChemBL_pickle/")] + \
    #        ["/data/lrl/QM_sym_pickles/" + i for i in os.listdir("/data/lrl/QM_sym_pickles/")]

    # data = pickle.load(open("/data/lrl/NAS_503_share/BindingDB_pickle/97.pkl", "rb")) + \
    #        pickle.load(open("/data/lrl/NAS_503_share/ChemBL_pickle/99.pkl", "rb")) + \
    #        pickle.load(open("/data/lrl/QM_sym_pickles/QM_sym_pickle_027.pickle", "rb"))
    data_qmb = pickle.load(open("/data/lrl/QM_sym_pickles/QM_sym_pickle_027.pickle", "rb"))
    data_bdb = pickle.load(open("/data/lrl/NAS_503_share/BindingDB_pickle/97.pkl", "rb"))
    random.shuffle(data_qmb)
    random.shuffle(data_bdb)

    # loss_qmb, record_qmb = test(None, calc, data_qmb, 1000)
    # loss_bdb, record_bdb = test(None, calc, data_bdb, 1000)
    # with open("./testing_process/20221210_tuned_{}_{}_{}.txt".format(str(round(loss_qmb,2)).rjust(10, "0"), str(round(loss_bdb,2)).rjust(10, "0"), "vanilla"), "w+") as f:
    #     f.write("### QM_SYM ###" + "\n")
    #     for r in record_qmb:
    #         f.write(str(r[0]) + "\t" + str(r[1]) + "\n")
    #     f.write("\n### BindingDB ###" + "\n")
    #     for r in record_bdb:
    #         f.write(str(r[0]) + "\t" + str(r[1]) + "\n")

    weights = ["./weight/" + i for i in os.listdir("./weight/") if i.startswith("20221207_self_trained_6")]
    for i, weight in tqdm(enumerate(weights), ncols=80):
        calc.spookynet = torch.load(weight)
        loss_qmb, record_qmb = test(None, calc, data_qmb, 1000)
        loss_bdb, record_bdb = test(None, calc, data_bdb, 1000)
        with open("./testing_process/20221210_trained_{}_{}_{}.txt".format(str(round(loss_qmb,2)).rjust(10, "0"), str(round(loss_bdb,2)).rjust(10, "0"), weight.split("/")[-1]), "w+") as f:
            f.write("### QM_SYM ###" + "\n")
            for r in record_qmb:
                f.write(str(r[0]) + "\t" + str(r[1]) + "\n")
            f.write("\n### BindingDB ###" + "\n")
            for r in record_bdb:
                f.write(str(r[0]) + "\t" + str(r[1]) + "\n")

    # """Find the best and worst ones."""
    # records = os.listdir("./testing_process/")
    # losses = [float(r.split("_")[2]) for r in records]
    # best, worst = "_".join(records[np.argmin(losses)].split("_")[3:]), "_".join(records[np.argmax(losses)].split("_")[3:])