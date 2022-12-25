import torch

from ase import Atoms
from spookynet import SpookyNetCalculator

# create ASE Atoms object for carbene
carbene = Atoms('CH2', positions=[
	( 0.000,  0.000,  0.000), 
	(-0.865, -0.584,  0.000), 
	( 0.865, -0.584,  0.000), 
])

# magmom is the number of unpaired electrons, i.e. 0 means singlet
carbene.set_calculator(SpookyNetCalculator(load_from="parameters.pth", charge=0, magmom=0))
print("singlet energy")
print(carbene.get_potential_energy())

# magmom=2 means 2 unpaired electrons => triplet state
carbene.set_calculator(SpookyNetCalculator(load_from="parameters.pth", charge=0, magmom=2))
print("triplet energy")
print(carbene.get_potential_energy())

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

calc = SpookyNetCalculator(load_from="parameters.pth", charge=0, magmom=0)
calc.spookynet.train()

calc._update_neighborlists(carbene)
Z = torch.from_numpy(carbene.get_atomic_numbers()).long().to(calc.spookynet.device)
Q = torch.Tensor([0]).float().to(calc.spookynet.device)
S = torch.Tensor([0]).float().to(calc.spookynet.device)
R = torch.from_numpy(carbene.get_positions()).float().to(calc.spookynet.device)
idx_i = calc.idx_i.long().to(calc.spookynet.device)
idx_j = calc.idx_j.long().to(calc.spookynet.device)

energy, f, ea, qa, ea_rep, ea_ele, ea_vdw, pa, c6 = calc.spookynet.energy(Z, Q, S, R, idx_i, idx_j)