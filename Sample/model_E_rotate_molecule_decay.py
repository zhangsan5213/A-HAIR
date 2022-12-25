'''
The modified GOODLE model to the finite case.
The input file structure is kept the same as the periodic one from the original GOODLE for convenience.
'''
import os
import time
import pickle
import timeit
import argparse
import platform
import itertools
import numpy as np
import matplotlib.pyplot as plt

from scipy import spatial
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class H2E(torch.nn.Module):
    def __init__(self):
        super(H2E, self).__init__()
        self.m0 = torch.nn.AdaptiveMaxPool2d((128, 128))
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=130,
                            kernel_size=3,
                            stride=1,
                            padding=2),
            torch.nn.BatchNorm2d(130),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(130, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(32 * 32 * 64, 10)
        self.mlp2 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.m0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


class spatial_transform(nn.Module):
    def __init__(self):
        super(spatial_transform, self).__init__()
        self.fc_loc = nn.Sequential(
            nn.Linear(24,48),
            nn.LeakyReLU(True),
            nn.Linear(48, 4),
            nn.LeakyReLU(True)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0, 1, 0, 0], dtype=torch.float))

    def forward(self, x):
        # transform the input
        x = self.fc_loc(x.reshape(-1, 24))
        return x


class orbital_potential(nn.Module):
    def __init__(self, type1="s", type2="px", sphere=None, device='cpu'):
        super(orbital_potential, self).__init__()
        function_dict = {"1s": self.orbital_1s,
                         "2s": self.orbital_2s,
                         "sp30": self.orbital_sp3_0,
                         "sp31": self.orbital_sp3_1,
                         "sp32": self.orbital_sp3_2,
                         "sp33": self.orbital_sp3_3,}

        self.orb_map1 = function_dict[type1]
        self.orb_map2 = function_dict[type2]
        self.sphere = sphere
        self.device = device
        self.a0 = nn.Parameter(torch.tensor([6.7], dtype=torch.float64, device=self.device))
        self.wf2ene = nn.Sequential(nn.Linear(self.sphere.shape[0]*2, 20), nn.ReLU(), nn.Linear(20, 1)).to(self.device)

    def orbital_1s(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (2 - r/self.a0)* torch.exp(- r/self.a0/2)
        wf = F.normalize(wf, 2, 1)
        return wf

    def orbital_2s(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (2 - r/self.a0)* torch.exp(- r/self.a0/2)
        wf = F.normalize(wf, 2, 1)
        return wf
    def orbital_px(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (position[:, :, 0] / self.a0) * torch.exp(- r / self.a0 / 2)
        wf = F.normalize(wf, 2, 1)
        return wf
    def orbital_py(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (position[:, :, 1] / self.a0) * torch.exp(- r / self.a0 / 2)
        wf = F.normalize(wf, 2, 1)
        return wf
    def orbital_pz(self, position):
        r = (position[:, :, 0] ** 2 + position[:, :, 1] ** 2 + position[:, :, 2] ** 2) ** 0.5
        wf = (position[:, :, 2] / self.a0) * torch.exp(- r / self.a0 / 2)
        wf = F.normalize(wf, 2, 1)
        return wf

    def orbital_sp3_0(self,position):
        return (self.orbital_2s(position) + self.orbital_px(position) + self.orbital_py(position) + self.orbital_pz(position))/2
    def orbital_sp3_1(self,position):
        return (self.orbital_2s(position) + self.orbital_px(position) - self.orbital_py(position) - self.orbital_pz(position))/2
    def orbital_sp3_2(self,position):
        return (self.orbital_2s(position) - self.orbital_px(position) + self.orbital_py(position) - self.orbital_pz(position))/2
    def orbital_sp3_3(self,position):
        return (self.orbital_2s(position) - self.orbital_px(position) - self.orbital_py(position) + self.orbital_pz(position))/2

    def forward(self, x):
        ## num_k, 4 * num_atoms, 4 * num_atoms, 7 (theta, vx, vy, vz, dist_xyz)
        if x.shape[1] == 0:
            return torch.zeros(x.shape[0], 0, 1, device=self.device)
        theta = x[:, :, 0]
        vx = x[:, :, 1]
        vy = x[:, :, 2]
        vz = x[:, :, 3]
        # construct the rotating matrix
        rotating_matrix = torch.zeros(x.shape[1], 3, 3).to(self.device)
        rotating_matrix[:, 0, 0] = torch.cos(theta) + (1 - torch.cos(theta)) * vx ** 2
        rotating_matrix[:, 0, 1] = (1 - torch.cos(theta)) * vx * vy - vz * torch.sin(theta)
        rotating_matrix[:, 0, 2] = (1 - torch.cos(theta)) * vx * vz + vy * torch.sin(theta)
        rotating_matrix[:, 1, 0] = (1 - torch.cos(theta)) * vx * vy + vz * torch.sin(theta)
        rotating_matrix[:, 1, 1] = torch.cos(theta) + (1 - torch.cos(theta)) * vy ** 2
        rotating_matrix[:, 1, 2] = (1 - torch.cos(theta)) * vy * vz - vx * torch.sin(theta)
        rotating_matrix[:, 2, 0] = (1 - torch.cos(theta)) * vx * vz - vy * torch.sin(theta)
        rotating_matrix[:, 2, 1] = (1 - torch.cos(theta)) * vy * vz + vx * torch.sin(theta)
        rotating_matrix[:, 2, 2] = torch.cos(theta) + (1 - torch.cos(theta)) * vz ** 2

        atom1_sphere = torch.zeros(x.shape[1], self.sphere.shape[0], 3).to(self.device)
        atom1_sphere[:, :, :] = self.sphere[None, :, :]
        atom1_sphere = torch.einsum("ijk,imk ->imj", rotating_matrix, atom1_sphere)
        atom1_sphere = self.orb_map1(atom1_sphere)

        atom2_sphere = torch.zeros(x.shape[1], self.sphere.shape[0], 3).to(self.device)
        atom2_sphere[:, :, :] = self.sphere[None, :, :]
        atom2_sphere[:, :, :] += x[:, :, 4:].squeeze(0)[:, None, :]
        atom2_sphere = torch.einsum("ijk,imk ->imj", rotating_matrix, atom2_sphere)
        atom2_sphere = self.orb_map2(atom2_sphere)

        grid_points = torch.cat((atom1_sphere, atom2_sphere), dim=1).to(self.device)
        energy = self.wf2ene( grid_points )
        return energy


class GOODLE_E(nn.Module):
    def __init__(self, device):
        super(GOODLE_E, self).__init__()
        radius = 3
        grid_interval = 1
        xyz = np.arange(-radius, radius + 1e-3, grid_interval)
        self.sphere = [[x, y, z] for x in xyz for y in xyz for z in xyz
                       if (x ** 2 + y ** 2 + z ** 2 <= radius ** 2) and [x, y, z] != [0, 0, 0]]
        self.sphere = torch.tensor(self.sphere).to(device)
        self.device = device
        self.min_dist = 1.6  # c-c for 1.54
        self.decay_rate = torch.nn.Parameter(torch.Tensor([1]).cuda()) ## for exponentially decayed bonding energy

        self.E_1s_1s = orbital_potential(type1="1s", type2="1s", sphere=self.sphere, device=self.device)

        self.E_sp30_sp30 = orbital_potential(type1="sp30", type2="sp30", sphere=self.sphere, device=self.device)
        self.E_sp30_sp31 = orbital_potential(type1="sp30", type2="sp31", sphere=self.sphere, device=self.device)
        self.E_sp30_sp32 = orbital_potential(type1="sp30", type2="sp32", sphere=self.sphere, device=self.device)
        self.E_sp30_sp33 = orbital_potential(type1="sp30", type2="sp33", sphere=self.sphere, device=self.device)
        self.E_sp31_sp30 = orbital_potential(type1="sp31", type2="sp30", sphere=self.sphere, device=self.device)
        self.E_sp31_sp31 = orbital_potential(type1="sp31", type2="sp31", sphere=self.sphere, device=self.device)
        self.E_sp31_sp32 = orbital_potential(type1="sp31", type2="sp32", sphere=self.sphere, device=self.device)
        self.E_sp31_sp33 = orbital_potential(type1="sp31", type2="sp33", sphere=self.sphere, device=self.device)
        self.E_sp32_sp30 = orbital_potential(type1="sp32", type2="sp30", sphere=self.sphere, device=self.device)
        self.E_sp32_sp31 = orbital_potential(type1="sp32", type2="sp31", sphere=self.sphere, device=self.device)
        self.E_sp32_sp32 = orbital_potential(type1="sp32", type2="sp32", sphere=self.sphere, device=self.device)
        self.E_sp32_sp33 = orbital_potential(type1="sp32", type2="sp33", sphere=self.sphere, device=self.device)
        self.E_sp33_sp30 = orbital_potential(type1="sp33", type2="sp30", sphere=self.sphere, device=self.device)
        self.E_sp33_sp31 = orbital_potential(type1="sp33", type2="sp31", sphere=self.sphere, device=self.device)
        self.E_sp33_sp32 = orbital_potential(type1="sp33", type2="sp32", sphere=self.sphere, device=self.device)
        self.E_sp33_sp33 = orbital_potential(type1="sp33", type2="sp33", sphere=self.sphere, device=self.device)

        self.E_1s_sp30 = orbital_potential(type1="1s", type2="sp30", sphere=self.sphere, device=self.device)
        self.E_1s_sp31 = orbital_potential(type1="1s", type2="sp31", sphere=self.sphere, device=self.device)
        self.E_1s_sp32 = orbital_potential(type1="1s", type2="sp32", sphere=self.sphere, device=self.device)
        self.E_1s_sp33 = orbital_potential(type1="1s", type2="sp33", sphere=self.sphere, device=self.device)
        self.E_sp30_1s = orbital_potential(type1="sp30", type2="1s", sphere=self.sphere, device=self.device)
        self.E_sp31_1s = orbital_potential(type1="sp31", type2="1s", sphere=self.sphere, device=self.device)
        self.E_sp32_1s = orbital_potential(type1="sp32", type2="1s", sphere=self.sphere, device=self.device)
        self.E_sp33_1s = orbital_potential(type1="sp33", type2="1s", sphere=self.sphere, device=self.device)

        self.E_1s =   nn.Parameter(torch.tensor([-4.75], dtype=torch.float64, device=self.device), requires_grad=False)
        self.E_2s =   nn.Parameter(torch.tensor([-2.99], dtype=torch.float64, device=self.device), requires_grad=False)
        self.E_2p =   nn.Parameter(torch.tensor([ 3.71], dtype=torch.float64, device=self.device), requires_grad=False)
        self.E_2sp3 = nn.Parameter(torch.tensor([ 1.48], dtype=torch.float64, device=self.device), requires_grad=False)
        self.CNN_layer = H2E().to(device)

        self.spatial_transform_sp0 = spatial_transform().to(device)
        self.spatial_transform_sp1 = spatial_transform().to(device)
        self.spatial_transform_sp2 = spatial_transform().to(device)
        self.spatial_transform_sp3 = spatial_transform().to(device)

    def forward(self, positions, atomic_numbers, distance_matrix,
                      ori_dist, dist_xs, dist_ys, dist_zs,
                      kRs, accessible_R, ks):
        num_atoms = len(positions)  # the positions for all atoms
        # order: s/px/py/pz
        num_accessible_R = len(accessible_R)
        num_k = len(ks)

        hydrogen_mask = [[int(atomic_number>1)]*4 for atomic_number in atomic_numbers] ## 0 for hydrogen, i.e. no sp3 hybridization
        hydrogen_mask = list(itertools.chain(*hydrogen_mask))
        hybrid_mask = np.array([str(hydrogen_mask[j[0]]) + str(hydrogen_mask[j[1]]) for j in itertools.product([i for i in range(num_atoms)], repeat=2)]).reshape(num_atoms, num_atoms)
        ## 1 if 2sp3, 0 if 1s.

        distance_matrices = torch.tensor(distance_matrix, dtype=torch.float32, device=self.device)
        ori_dist = torch.tensor(ori_dist, dtype=torch.float32, device=self.device)
        kRs = torch.tensor(kRs, dtype=torch.float32, device=self.device)
        dist_xs = torch.tensor(dist_xs, dtype=torch.float32, device=self.device)
        dist_ys = torch.tensor(dist_ys, dtype=torch.float32, device=self.device)
        dist_zs = torch.tensor(dist_zs, dtype=torch.float32, device=self.device)
        
        H_real = torch.zeros(num_k, 4 * num_atoms, 4 * num_atoms, device=self.device)
        H_imag = torch.zeros(num_k, 4 * num_atoms, 4 * num_atoms, device=self.device)
        
        bond_decay_factors = torch.exp(-ori_dist.unsqueeze(0).repeat(num_k, 1, 1)/self.decay_rate)

        mask_sp3_sp3 = torch.tensor([[ 1,  2,  3,  4],
                                     [ 5,  6,  7,  8],
                                     [ 9, 10, 11, 12],
                                     [13, 14, 15, 16]], dtype=torch.float32, device=self.device)
        mask_sp0_sp3 = torch.tensor([[17, 18, 19, 20],
                                     [ 0,  0,  0,  0],
                                     [ 0,  0,  0,  0],
                                     [ 0,  0,  0,  0]], dtype=torch.float32, device=self.device)
        mask_sp3_sp0 = torch.tensor([[21,  0,  0,  0],
                                     [22,  0,  0,  0],
                                     [23,  0,  0,  0],
                                     [24,  0,  0,  0]], dtype=torch.float32, device=self.device)
        mask_sp0_sp0 = torch.tensor([[25,  0,  0,  0],
                                     [ 0,  0,  0,  0],
                                     [ 0,  0,  0,  0],
                                     [ 0,  0,  0,  0]], dtype=torch.float32, device=self.device)
        # mask = torch.repeat_interleave(mask.unsqueeze(0), num_atoms, dim=0).reshape(4 * num_atoms, -1)
        # mask = torch.repeat_interleave(mask.unsqueeze(1), num_atoms, dim=0).reshape(4 * num_atoms, -1)
        ## This is for different kinds of overlapping, but for n=2/sp3 only.
        ## n=1/s is expected to be added for hydrogen. A for loop can be necessary?
        ## 09 for 1s-1s, 10 for 1s-2sp3_0, 11 for 1s-2sp3_1, 12 for 1s-2sp3_2, 13 for 1s-2sp3_3, etc.
        mask = mask_sp3_sp3.unsqueeze(0).unsqueeze(0).repeat(num_atoms, num_atoms, 1, 1)
        mask[(hybrid_mask == "00"), :, :] = mask_sp0_sp0
        mask[(hybrid_mask == "01"), :, :] = mask_sp0_sp3
        mask[(hybrid_mask == "10"), :, :] = mask_sp3_sp0

        mask = mask.split(1, dim=0)
        mask = [torch.cat(list(mask_split.squeeze(0).split(1, dim=0)), dim=2).squeeze(0) for mask_split in mask]
        mask = torch.cat(mask, dim=0)

        H_real[:, (mask == 1) & (ori_dist < 0.01)] = self.E_2sp3
        H_imag[:, (mask == 1) & (ori_dist < 0.01)] = 0
        H_real[:, (mask == 6) & (ori_dist < 0.01)] = self.E_2sp3
        H_imag[:, (mask == 6) & (ori_dist < 0.01)] = 0
        H_real[:, (mask ==11) & (ori_dist < 0.01)] = self.E_2sp3
        H_imag[:, (mask ==11) & (ori_dist < 0.01)] = 0
        H_real[:, (mask ==16) & (ori_dist < 0.01)] = self.E_2sp3
        H_imag[:, (mask ==16) & (ori_dist < 0.01)] = 0
        H_real[:, (mask ==25) & (ori_dist < 0.01)] = self.E_1s
        H_imag[:, (mask ==25) & (ori_dist < 0.01)] = 0

        # info matrix: (wavevector) kx,ky,kz (dist) x,y,z,r (orbital) x,y,z
        info = torch.zeros(num_k, 4 * num_atoms, 4 * num_atoms, 7, dtype=torch.float32, device=self.device)
        # the first four elements are the rotating parameters.
        k = 8
        input_for_spatial = torch.zeros(num_k, 4 * num_atoms, k, 3, dtype=torch.float32, device=self.device)
        ## The dimensions of input_for_spatial refer to:
        ## num_k: number of k points;
        ## 4*num_atoms: total number of orbitals;
        ## k: number of bands;
        ## 3: coordinates.
        
        value, idx1 = torch.sort(torch.tensor(distance_matrix[13]))
        value, idx2 = torch.sort(idx1)
        distance_matrix[0][idx2 < 8].reshape(4 * num_atoms, k)
        input_for_spatial[:, :, :, 0] = dist_xs[13][idx2 < 8].reshape(4 * num_atoms, k)
        input_for_spatial[:, :, :, 1] = dist_ys[13][idx2 < 8].reshape(4 * num_atoms, k)
        input_for_spatial[:, :, :, 2] = dist_zs[13][idx2 < 8].reshape(4 * num_atoms, k)

        # result: theta, vx, vy, vz
        rotate_sp0 = self.spatial_transform_sp0(input_for_spatial[:, ::4, :, :].double())
        rotate_sp1 = self.spatial_transform_sp1(input_for_spatial[:, ::4, :, :].double())
        rotate_sp2 = self.spatial_transform_sp2(input_for_spatial[:, ::4, :, :].double())
        rotate_sp3 = self.spatial_transform_sp3(input_for_spatial[:, ::4, :, :].double())
        rotate = torch.cat([rotate_sp0.unsqueeze(0),
                            rotate_sp1.unsqueeze(0),
                            rotate_sp2.unsqueeze(0),
                            rotate_sp3.unsqueeze(0)], dim=0)
        rotate = torch.flatten(rotate.permute(1, 0, 2), start_dim=0, end_dim=1)
        ## each four in "rotate" comes from each one in the "rotate_spx"

        # # ## make the ones with 1s orbitals (0,1,0,0), so that it would affect the sp3 rotations
        for i in range(len(hydrogen_mask)):
            if hydrogen_mask[i] == 0:
                rotate[i, :] = torch.tensor([0, 1, 0, 0])
        # # # print(atomic_numbers)
        # # # print(rotate)
        # # # exit()

        info[:, :, :, :4] = rotate[None, None, :, :]
        
        # matrix form to estimate the Hamiltonian
        for i in range(num_accessible_R):
            ## for each point, add into the Hamiltonian on the index by calculation of the point
            tmp_kR = kRs[i]
            factor_real = torch.cos(tmp_kR)
            factor_imag = torch.sin(tmp_kR)
            # E_xx or E_ss, same direction xx or same direction ss
            # step 2: assign the off-diagonal elements. all are in type xy
            # E_xy, different direction xy.
            info[:, :, :, 4] = torch.abs(dist_xs[i])
            info[:, :, :, 5] = torch.abs(dist_ys[i])
            info[:, :, :, 6] = torch.abs(dist_zs[i])  # these are the absolute value.
            mask_dist = (distance_matrices[i] < self.min_dist) & (distance_matrices[i] > 0.01) ## the orbitals that considered bonded

            indx = mask_dist & (mask == 1)
            energy_func = self.E_sp30_sp30(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask == 2)
            energy_func = self.E_sp30_sp31(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask == 3)
            energy_func = self.E_sp30_sp32(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask == 4)
            energy_func = self.E_sp30_sp33(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]


            indx = mask_dist & (mask == 5)
            energy_func = self.E_sp31_sp30(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask == 6)
            energy_func = self.E_sp31_sp31(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask == 7)
            energy_func = self.E_sp31_sp32(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask == 8)
            energy_func = self.E_sp31_sp33(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            

            indx = mask_dist & (mask == 9)
            energy_func = self.E_sp32_sp30(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==10)
            energy_func = self.E_sp32_sp31(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==11)
            energy_func = self.E_sp32_sp32(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==12)
            energy_func = self.E_sp32_sp33(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            

            indx = mask_dist & (mask ==13)
            energy_func = self.E_sp33_sp30(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==14)
            energy_func = self.E_sp33_sp31(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==15)
            energy_func = self.E_sp33_sp32(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==16)
            energy_func = self.E_sp33_sp33(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            

            indx = mask_dist & (mask ==17)
            energy_func = self.E_1s_sp30(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==18)
            energy_func = self.E_1s_sp31(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==19)
            energy_func = self.E_1s_sp32(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==20)
            energy_func = self.E_1s_sp33(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            

            indx = mask_dist & (mask ==21)
            energy_func = self.E_sp30_1s(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==22)
            energy_func = self.E_sp31_1s(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==23)
            energy_func = self.E_sp32_1s(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            
            indx = mask_dist & (mask ==24)
            energy_func = self.E_sp33_1s(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            

            indx = mask_dist & (mask ==25)
            energy_func = self.E_1s_1s(info[:, indx, :].double()).squeeze(-1)
            H_real[:, indx] += energy_func * factor_real[:, indx]
            # H_imag[:, indx] += energy_func * factor_imag[:, indx]
            

        real_H = torch.cat((torch.cat((H_real * bond_decay_factors,-H_imag * bond_decay_factors), dim=2),
                            torch.cat((H_imag * bond_decay_factors, H_real * bond_decay_factors), dim=2)), dim=1)
        enes = torch.sum(real_H).reshape(1,1) #do not ask about the dimension

        return -enes