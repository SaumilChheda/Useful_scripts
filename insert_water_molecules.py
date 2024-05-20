#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import pandas as pd
import os
import random

def get_dist_from_all_atoms(atom_i_coords, coords, lattice):
    
    dist_ij = np.zeros([np.shape(coords)[0],4])
    new_coords_red = np.zeros([np.shape(coords)[0],3])
    
    coords_red = np.transpose(lattice_inv.dot(np.transpose(coords)))
    
    atom_i_coords_red = np.transpose(lattice_inv.dot(np.transpose(atom_i_coords)))
    
    for j in range(0,np.shape(coords_red)[0]):
        atom_j_coords = coords_red[j,:]
        
        dist_ij[j,0] = np.linalg.norm(atom_j_coords-atom_i_coords_red)
        
        if dist_ij[j,0] > 0.5:
            for x in range(-1,2):
                for y in range(-1,2):
                    for z in range(-1,2):
                        coords_image = atom_j_coords + [[x,y,z]]
                        dist_image = np.linalg.norm(coords_image-atom_i_coords_red)
                    
                        if dist_image < dist_ij[j,0]:
                            dist_ij[j,0] = dist_image
                            atom_j_coords = coords_image
                        
        new_coords_red[j,:] = atom_j_coords
    
    dist_ij[:,1:4] = np.transpose(np.transpose(lattice).dot(np.transpose(new_coords_red)))
        
    return dist_ij

def get_Hcoord(Ow_site_coord, Hbond_dir, lattice, OHw_len):
    
    x = Hbond_dir[0]
    y = Hbond_dir[1]
    z = Hbond_dir[2]
    
    cos_beta = np.sqrt(x*x+y*y)/np.sqrt(x*x+y*y+z*z)
    beta = np.arccos(cos_beta)
    
    if z<0:
        beta = -beta

    rot_mat1 = np.asarray([[np.cos(beta), 0, np.sin(beta)],
               [0, 1, 0],
               [-np.sin(beta), 0, np.cos(beta)]])

    cos_alpha = y/np.sqrt(x*x+y*y)
    alpha = np.arccos(cos_alpha)

    if x<0:
        alpha = -alpha

    rot_mat2 = np.asarray([[np.cos(alpha), -np.sin(alpha), 0],
               [np.sin(alpha), np.cos(alpha), 0],
               [0, 0, 1]])
    
    OHcoord_ref = np.transpose(rot_mat2.dot(rot_mat1.dot(np.transpose(Hbond_dir))))
    
    deg_var = random.random()
    
    if random.random() > 0.5:
        deg_var = -deg_var
    
    HOH = np.radians(-70.2 + deg_var*10)
    
    OHcoord_rot = np.asarray([[np.cos(HOH), -np.sin(HOH), 0],
               [np.sin(HOH), np.cos(HOH), 0],
               [0, 0, 1]])@OHcoord_ref
    
    gamma = np.arccos(random.random())
    
    if random.random() > 0.5:
        gamma = -gamma
    
    OHcoord_rot = np.asarray([[1, 0, 0],
               [0, np.cos(gamma), -np.sin(gamma)],
               [0, np.sin(gamma), np.cos(gamma)]])@OHcoord_ref
    
    rot_mat2_inv = np.linalg.inv(rot_mat2)
    rot_mat1_inv = np.linalg.inv(rot_mat1)
    
    OHcoord_rot = np.transpose(rot_mat2_inv.dot(np.transpose(OHcoord_rot)))
    OHcoord_rot = np.transpose(rot_mat1_inv.dot(np.transpose(OHcoord_rot)))
    
    OHcoord_rot = OHcoord_rot/np.linalg.norm(OHcoord_rot)
    
    Hcoord = OHcoord_rot*OHw_len + Ow_site_coord
    
    return Hcoord

def Hcoord_check(H1_coord, Ow_site_coord, Hbond_dir, coords, lattice, rmin):
    
    Hcoord_ij_data = get_dist_from_all_atoms(H1_coord, coords, lattice)
    
    nn1_index = np.argmin(Hcoord_ij_data[:,0])
    
    Hcoord_nn1_dist = np.linalg.norm(Hcoord_ij_data[nn1_index,1:4]-H1_coord)
    
    if Hcoord_nn1_dist < rmin:
        
        Hbond_dir = Ow_site_coord - H1_coord
        Hbond_dir = Hbond_dir/np.linalg.norm(Hbond_dir)
        Hbond_dir = np.reshape(Hbond_dir,[3,])
        
        H1_coord_check = 1
    
    else:
        
        Hbond_dir = Hbond_dir
        
        H1_coord_check = 0
        
    return Hbond_dir, H1_coord_check

def Hangle_check(H2_coord, Ow_site_coord, H1_coord):
    
    OH1 = H1_coord - Ow_site_coord
    OH1 = OH1/np.linalg.norm(OH1)
    
    OH2 = H2_coord - Ow_site_coord
    OH2 = OH2/np.linalg.norm(OH2)
    
    HOH = np.degrees(np.arccos(np.dot(OH1, np.transpose(OH2))))
    
    if HOH < 120 and HOH > 95:
        H2_angle_check = 0
    else:
        H2_angle_check = 1
    
    return H2_angle_check

#Currently only handles P21_c symmetry (space group no. 4)

def get_symmetry(Ow_site_coord, H1_coord, H2_coord, lattice):
    
    symm_coords = np.zeros([3,9])
    
    #Symmetry points are:
    #x, y, z
    #-x, -y, -z
    #-x, y+1/2, -z+1/2
    #x, -y+1/2, z+1/2
    
    coords_water = np.append(np.append(Ow_site_coord, H1_coord, axis=0), H2_coord, axis=0)
    
    coords_symm1_red = np.zeros([3,3])
    coords_symm2_red = np.zeros([3,3])
    coords_symm3_red = np.zeros([3,3])
    
    lattice_inv = np.linalg.inv(np.transpose(lattice))
    
    coords_water_red = np.transpose(lattice_inv.dot(np.transpose(coords_water)))
    
    coords_symm1_red[:,:] = -coords_water_red[:,:]
    
    coords_symm2_red[:,0] = -coords_water_red[:,0]
    coords_symm2_red[:,1] = coords_water_red[:,1] + np.asarray([1/2,1/2,1/2])
    coords_symm2_red[:,2] = -coords_water_red[:,2] + np.asarray([1/2,1/2,1/2])
    
    coords_symm3_red[:,0] = coords_water_red[:,0]
    coords_symm3_red[:,1] = -coords_water_red[:,1] + np.asarray([1/2,1/2,1/2])
    coords_symm3_red[:,2] = coords_water_red[:,2] + np.asarray([1/2,1/2,1/2])

    for i in range(0,3):
        for j in range(0,3):
            coords_symm1_red[i,j] = coords_symm1_red[i,j] - np.floor(coords_symm1_red[i,j])
            coords_symm2_red[i,j] = coords_symm2_red[i,j] - np.floor(coords_symm2_red[i,j])
            coords_symm3_red[i,j] = coords_symm3_red[i,j] - np.floor(coords_symm3_red[i,j])
    
    coords_symm1 = np.transpose(np.transpose(lattice).dot(np.transpose(coords_symm1_red)))
    coords_symm2 = np.transpose(np.transpose(lattice).dot(np.transpose(coords_symm2_red)))
    coords_symm3 = np.transpose(np.transpose(lattice).dot(np.transpose(coords_symm3_red)))
    
    symm_coords[0,:] = np.reshape(coords_symm1, [1,9])
    symm_coords[1,:] = np.reshape(coords_symm2, [1,9])
    symm_coords[2,:] = np.reshape(coords_symm3, [1,9])
    
    return symm_coords

def gen_POSCAR(water_unique_site_coords_full_indv, coords_all, lattice, atom_label_all, natoms, n_unique_site, struc_i_curr, substruc_i_curr):
    
    #lattice_inv = np.linalg.inv(np.transpose(lattice))
    
    os.system("mkdir site_" + str(n_unique_site))
    
    os.chdir("./site_" + str(n_unique_site))
    os.system("pwd") 
    f = open("POSCAR", 'w')
    f.close()

    f = open("POSCAR", 'a')
    f.write("site_" + str(n_unique_site) + "\n")
    f.write(str(1.0) + "\n")
    f.write(str(lattice[0,0]) + "    " + str(lattice[0,1]) + "    " + str(lattice[0,2]) + "\n")
    f.write(str(lattice[1,0]) + "    " + str(lattice[1,1]) + "    " + str(lattice[1,2]) + "\n")
    f.write(str(lattice[2,0]) + "    " + str(lattice[2,1]) + "    " + str(lattice[2,2]) + "\n")

    natoms_temp = np.zeros(np.shape(natoms))
    natoms_temp[0,:] = natoms[0,:]
    
    index_O = list(atom_label_all[0,:]).index('O')
    index_H = list(atom_label_all[0,:]).index('H')
    n_water = int(np.shape(water_unique_site_coords_full_indv)[0])
    
    natoms_temp[0,index_O] = int(natoms_temp[0,index_O] + n_water)
    natoms_temp[0,index_H] = int(natoms_temp[0,index_H] + 2*n_water)
    
    for i in range(0,np.shape(atom_label_all)[1]):
        element = atom_label_all[0,i]
        f.write("  " + str(element) + "  ")
    f.write("\n")

    for i in range(0,np.shape(atom_label_all)[1]):
        f.write("  " + str(int(natoms_temp[0,i])) + "  ")
    f.write("\n")

    f.write("Cartesian" + "\n")
    #f.write("Direct" + "\n")
    
    natoms_temp = np.append([[0]], natoms, axis=1)
    
    for i in range(0,np.shape(natoms)[1]):
        coords_element = np.zeros([natoms[0,i],3])
        
        coords_element = coords_all[np.sum(natoms[0,0:i]):np.sum(natoms[0,0:i+1]),:]
        
        if atom_label_all[0,i] == 'O':
            coords_element = np.append(coords_element, water_unique_site_coords_full_indv[:,0:3], axis=0)
            
        if atom_label_all[0,i] == 'H':
            coords_element = np.append(coords_element, water_unique_site_coords_full_indv[:,3:6], axis=0)
            coords_element = np.append(coords_element, water_unique_site_coords_full_indv[:,6:9], axis=0)
        
        #coords_element_red = np.transpose(lattice_inv.dot(np.transpose(coords_element)))
        
        for j in range(0,np.shape(coords_element)[0]):
            f.write("  " + str(coords_element[j,0]) + "  " + str(coords_element[j,1]) + "  " + str(coords_element[j,2]) + "\n")

    f.close()
    
    os.system("cp -r ../../../sample_files/run1 run1")
    os.chdir("./run1")
    os.system("cp ../POSCAR .")
    os.system("rm POTCAR")
    
    os.system("touch POTCAR") 
    for i in range(0,np.shape(atom_label_all)[1]):
        element = atom_label_all[0,i]
        os.system("cat ../../../../POTCAR_library/potpaw_PBE.52/" + str(element) + "/POTCAR >> POTCAR")
    
    os.system("sed -i 's/MOF303analogue_1_substruc_1_site_1/MOF303analogue_" + str(struc_i_curr) + "_substruc_" + str(substruc_i_curr)+ "_site_" + str(n_unique_site) +  "/' jobvaspcont.sh")
    
#    os.system("sbatch jobvaspcont.sh")
    os.chdir("./..")
     
    os.chdir("./..")
    
    return None

# In[8]:


nbeads = np.asarray([1, 1, 1, 13])
atom_types_in_POSCAR = ["Al", "O", "C", "H", "N", "S"]

struc_i_begin = int(input("Enter the structure to begin with: "))
struc_i_end = int(input("Enter the structure to end at: "))
nstruc = int(struc_i_end-struc_i_begin+1)
cutoff = 1.7 #cutoff for unique sites
Hbond_len = 2.7 #dist at which to put the Ow
OHw_len = 0.96 #distance at which to put Hw
rmin = 1.2 #distance to check overlap
max_orient_try = 500

#os.system("cd linker_coordinates")

for struc_i in range(struc_i_begin,struc_i_end+1):
     
    new_structure_name=str(struc_i)
    os.chdir("./" + new_structure_name)

    #intially read the file as '"POSCAR_" + str(struc_i)'
    #Now CONTCAR

    os.system("pwd")
    os.system("ls")

    substruc_i_begin = int(input("Enter the SUB-structure to begin with: "))
    substruc_i_end = int(input("Enter the SUB-structure to end at: "))

    for substruc_i in range(substruc_i_begin,substruc_i_end+1):

        new_substructure_name=str(substruc_i)
        os.chdir("./struc_" + new_substructure_name)
   
        #intially read the file as '"POSCAR_" + str(struc_i)'
        #Now CONTCAR

        os.system("pwd")

        lattice = pd.read_csv("CONTCAR", header=None, skiprows=2, nrows=3, delim_whitespace=True).values
        
        lattice_inv = np.linalg.inv(np.transpose(lattice))
        
        atom_label_all = pd.read_csv("CONTCAR", header=None, skiprows=5, nrows=1, delim_whitespace=True).values
        
        natoms = pd.read_csv("CONTCAR", header=None, skiprows=6, nrows=1, delim_whitespace=True).values
        natoms_total = np.sum(natoms)
        
        coords_type_label = pd.read_csv("CONTCAR", header=None, skiprows=7, nrows=1, delim_whitespace=True).values[0,0][0:1].upper()
        
        coords = pd.read_csv("CONTCAR", header=None, skiprows=8, delim_whitespace=True).values
        
        if coords_type_label == "C":
            coords_red = np.transpose(lattice_inv.dot(np.transpose(coords)))
            for i in range(0,np.shape(coords_red)[0]):
                for j in range(0,3):
                    coords_red[i,j] = coords_red[i,j] - np.floor(coords_red[i,j])
        elif coords_type_label == "D":
            coords_red = np.zeros(np.shape(coords))
            coords_red = coords
        
        coords = np.transpose(np.transpose(lattice).dot(np.transpose(coords_red)))
        
        atom_label = np.chararray([natoms_total,1], itemsize=2)
        count = 0
        natom_types = np.shape(atom_label_all)[1]
        for i in range(0,natom_types):
            for j in range(0,natoms[0,i]):
                atom_label[count] = atom_label_all[0,i]
                count += 1
        atom_label = atom_label.decode('utf-8')
        
        n_unique_site = 0
        water_unique_site_coords = np.zeros([1,3]) #first 3 coords in a row for O, next 3 for H1 and next 3 for H2
        water_unique_site_coords_full = np.zeros([1,9])
        
        for i in range(0,natoms_total):
            print(i)
            if atom_label[i] in ['O', 'N', 'S', 'F', 'Cl', 'P']:
                atom_i_coords = np.zeros([1,3])
                atom_i_coords = coords[i]
                
                mod_ij_data = np.zeros([natoms_total,4])
                
                mod_ij_data = get_dist_from_all_atoms(atom_i_coords, coords, lattice)
                    
                mod_ij_data = np.delete(mod_ij_data, i, 0)
                atom_label_temp = np.delete(atom_label, i, 0)
                
                nn1_index = np.argmin(mod_ij_data[:,0])
                
                if atom_label_temp[nn1_index] == 'H' and np.linalg.norm(mod_ij_data[nn1_index,1:4] - atom_i_coords) < 1.1:
                    
                    atom1 = atom_i_coords
                    atom2 = mod_ij_data[nn1_index,1:4]
                    
                    Hbond_dir = atom2 - atom1
                    Hbond_dir = Hbond_dir/np.linalg.norm(Hbond_dir)
                    
                    Ow_site_coord = Hbond_dir*Hbond_len + atom_i_coords
                    Ow_site_coord = np.reshape(Ow_site_coord,[1,3])
                    
                    check_prev_site = 0
                    
                    if n_unique_site == 0:
                        water_unique_site_coords = Ow_site_coord
                        n_unique_site += 1
                        
                        H1_coord_check = 10
                        try_count_1 = 0
                        
                        while H1_coord_check != 0:
                            
                            if try_count_1 > max_orient_try:
                                break
                                
                            H1_coord = get_Hcoord(Ow_site_coord, Hbond_dir, lattice, OHw_len)
                        
                            Hbond_dir, H1_coord_check = Hcoord_check(H1_coord, Ow_site_coord, Hbond_dir, coords, lattice, rmin)
                            
                            try_count_1 += 1
                        
                        print("\t" + str(try_count_1))

                        Hbond_dir2 = Ow_site_coord - H1_coord
                        Hbond_dir2 = Hbond_dir2/np.linalg.norm(Hbond_dir2)
                        Hbond_dir2 = np.reshape(Hbond_dir2,[3,])
                        
                        H2_coord_check = 10
                        H2_angle_check = 10
                        
                        try_count_2 = 0
                        
                        while H2_coord_check != 0 or H2_angle_check != 0:
                            
                            if try_count_2 > max_orient_try:
                                break
                            
                            H2_coord = get_Hcoord(Ow_site_coord, Hbond_dir2, lattice, OHw_len)
                        
                            Hbond_dir2, H2_coord_check = Hcoord_check(H2_coord, Ow_site_coord, Hbond_dir2, np.append(coords,H1_coord, axis=0), lattice, rmin)
                            
                            H2_angle_check = Hangle_check(H2_coord, Ow_site_coord, H1_coord)
                            
                            try_count_2 += 1
                        
                        print("\t" + "\t" + str(try_count_2))
                        
                        water_unique_site_coords_full = np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1)
                        
                        symm_coords = get_symmetry(Ow_site_coord, H1_coord, H2_coord, lattice)
                        
                        water_unique_site_coords_full = np.append(water_unique_site_coords_full, symm_coords, axis=0)
                        
                        water_unique_site_coords = np.append(water_unique_site_coords, symm_coords[:,0:3], axis=0)

                        gen_POSCAR_dec = int(input("Do you want to generate input file (1 for Y and 0 for N): "))
                        
                        if gen_POSCAR_dec == 1:

                            os.system("touch struc_log.txt")
                            os.system("echo 'First number is n_unique_site number, second line is the try_count, third number is try_count for second H' >> struc_log.txt")
                            os.system("echo " + str(n_unique_site) + " >> struc_log.txt")
                            os.system("echo " + str(try_count_1) + " >> struc_log.txt")

                            if try_count_1 > max_orient_try:
                                os.system("echo 'WARNING' >> struc_log.txt")

                            os.system("echo " + str(try_count_2) + " >> struc_log.txt")

                            if try_count_2 > max_orient_try:
                                os.system("echo 'WARNING' >> struc_log.txt")

                            os.system("echo '****' >> struc_log.txt")

                            gen_POSCAR(np.append(np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1), symm_coords, axis=0), coords, lattice, atom_label_all, natoms, n_unique_site, struc_i, substruc_i)
                        
                    else:
                        for ows in range(0,np.shape(water_unique_site_coords)[0]):
                            
                            Ow_site_coord_red = np.transpose(lattice_inv.dot(np.transpose(Ow_site_coord)))
                            
                            for p in range(0,3):
                                Ow_site_coord_red[0,p] = Ow_site_coord_red[0,p] - np.floor(Ow_site_coord_red[0,p])
                                                    
                            Ow_site_coord = np.transpose(np.transpose(lattice).dot(np.transpose(Ow_site_coord_red)))                       
                            
                            if np.linalg.norm(water_unique_site_coords[ows,:]-Ow_site_coord) < cutoff:
                                check_prev_site = 1
                                            
                        if check_prev_site == 0:
                            water_unique_site_coords = np.append(water_unique_site_coords, Ow_site_coord, axis=0)
                            n_unique_site += 1
                            
                            H1_coord_check = 10
                            try_count_1 = 0
                            
                            while H1_coord_check != 0:
                                
                                if try_count_1 > max_orient_try:
                                    break
                                    
                                H1_coord = get_Hcoord(Ow_site_coord, Hbond_dir, lattice, OHw_len)
                        
                                Hbond_dir, H1_coord_check = Hcoord_check(H1_coord, Ow_site_coord, Hbond_dir, coords, lattice, rmin)
                                try_count_1 += 1
                            
                            print("\t" + str(try_count_1))

                            Hbond_dir2 = Ow_site_coord - H1_coord
                            Hbond_dir2 = Hbond_dir2/np.linalg.norm(Hbond_dir2)
                            Hbond_dir2 = np.reshape(Hbond_dir2,[3,])
                        
                            H2_coord_check = 10
                            H2_angle_check = 10
                            try_count_2 = 0
                            
                            while H2_coord_check != 0 or H2_angle_check != 0:
                                
                                if try_count_2 > max_orient_try:
                                    break

                                H2_coord = get_Hcoord(Ow_site_coord, Hbond_dir2, lattice, OHw_len)
                        
                                Hbond_dir2, H2_coord_check = Hcoord_check(H2_coord, Ow_site_coord, Hbond_dir2, np.append(coords,H1_coord, axis=0), lattice, rmin)
                            
                                H2_angle_check = Hangle_check(H2_coord, Ow_site_coord, H1_coord)
                                try_count_2 += 1
                                    
                            print("\t" + "\t" + str(try_count_2))
        
                            water_unique_site_coords_full = np.append(water_unique_site_coords_full, np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1), axis=0)
                            
                            symm_coords = get_symmetry(Ow_site_coord, H1_coord, H2_coord, lattice)
                        
                            water_unique_site_coords_full = np.append(water_unique_site_coords_full, symm_coords, axis=0)
                        
                            water_unique_site_coords = np.append(water_unique_site_coords, symm_coords[:,0:3], axis=0)

                            gen_POSCAR_dec = int(input("Do you want to generate input file (1 for Y and 0 for N): "))

                            if gen_POSCAR_dec == 1:

                                os.system("touch struc_log.txt")
                                os.system("echo 'First number is n_unique_site number, second line is the try_count, third number is try_count for second H' >> struc_log.txt")
                                os.system("echo " + str(n_unique_site) + " >> struc_log.txt")
                                os.system("echo " + str(try_count_1) + " >> struc_log.txt")

                                if try_count_1 > max_orient_try:
                                    os.system("echo 'WARNING' >> struc_log.txt")
                                
                                os.system("echo " + str(try_count_2) + " >> struc_log.txt")

                                if try_count_2 > max_orient_try:
                                    os.system("echo 'WARNING' >> struc_log.txt")

                                os.system("echo '****' >> struc_log.txt")

                                gen_POSCAR(np.append(np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1), symm_coords, axis=0), coords, lattice, atom_label_all, natoms, n_unique_site, struc_i, substruc_i)
                            
                if atom_label_temp[nn1_index] in ['C', 'O', 'N', 'S', 'F', 'Cl', 'P'] and np.linalg.norm(mod_ij_data[nn1_index,1:4] - atom_i_coords) < 1.6:
                    atom_label_temp_nn1 = 'X'
                    atom_label_temp_nn1 = atom_label_temp[nn1_index]
                    atom1 = atom_i_coords
                    atom2 = mod_ij_data[nn1_index,1:4]
                    
                    mod_ij_data = np.delete(mod_ij_data, nn1_index, 0)
                    atom_label_temp = np.delete(atom_label_temp, nn1_index, 0)
                    
                    nn2_index = np.argmin(mod_ij_data[:,0])
                    
                    if atom_label_temp[nn2_index] in ['C', 'O', 'N', 'S', 'F', 'Cl', 'P', 'H'] and np.linalg.norm(mod_ij_data[nn2_index,1:4] - atom_i_coords) < 1.6:
                       
                        if atom_label_temp[nn2_index] == 'H':
                            if atom_label_temp_nn1 != 'O':
                                continue

                        atom3 = mod_ij_data[nn2_index,1:4]
                        
                        bond_dir1 = atom2 - atom1
                        bond_dir1 = bond_dir1/np.linalg.norm(bond_dir1)
                        
                        bond_dir2 = atom3 - atom1
                        bond_dir2 = bond_dir2/np.linalg.norm(bond_dir2)
                        
                        Hbond_dir = -(bond_dir1 + bond_dir2)
                        Hbond_dir = Hbond_dir/np.linalg.norm(Hbond_dir)
                    
                        Ow_site_coord = Hbond_dir*Hbond_len + atom_i_coords
                        Ow_site_coord = np.reshape(Ow_site_coord,[1,3])
                    
                        check_prev_site = 0
                    
                        if n_unique_site == 0:
                            water_unique_site_coords = Ow_site_coord
                            n_unique_site += 1

                            H1_coord = -Hbond_dir*OHw_len + Ow_site_coord
                            
                            H2_coord = get_Hcoord(Ow_site_coord, Hbond_dir, lattice, OHw_len)
                            
                            H2_coord_check = 10
                            H2_angle_check = 10 
                            try_count_2 = 0
                            
                            while H2_coord_check != 0 or H2_angle_check != 0:

                                if try_count_2 > max_orient_try:
                                    break
                               
                                H2_coord = get_Hcoord(Ow_site_coord, Hbond_dir, lattice, OHw_len)
                        
                                Hbond_dir, H2_coord_check = Hcoord_check(H2_coord, Ow_site_coord, Hbond_dir, coords, lattice, rmin)

                                H2_angle_check = Hangle_check(H2_coord, Ow_site_coord, H1_coord)

                                try_count_2 += 1

                            print("\t" + str(1))
                            print("\t" + "\t" + str(try_count_2))

                            water_unique_site_coords_full = np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1)
                            
                            symm_coords = get_symmetry(Ow_site_coord, H1_coord, H2_coord, lattice)
                        
                            water_unique_site_coords_full = np.append(water_unique_site_coords_full, symm_coords, axis=0)
                        
                            water_unique_site_coords = np.append(water_unique_site_coords, symm_coords[:,0:3], axis=0)

                            gen_POSCAR_dec = int(input("Do you want to generate input file (1 for Y and 0 for N): "))

                            if gen_POSCAR_dec == 1:

                                os.system("touch struc_log.txt")
                                os.system("echo 'First number is n_unique_site number, second line is the try_count, third number is try_count for second H' >> struc_log.txt")
                                os.system("echo " + str(n_unique_site) + " >> struc_log.txt")
                                os.system("echo 'H1 fixed' >> struc_log.txt")
                                os.system("echo " + str(try_count_2) + " >> struc_log.txt")

                                if try_count_2 > max_orient_try:
                                    os.system("echo 'WARNING' >> struc_log.txt")

                                os.system("echo '****' >> struc_log.txt")

                                gen_POSCAR(np.append(np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1), symm_coords, axis=0), coords, lattice, atom_label_all, natoms, n_unique_site, struc_i,substruc_i)
                              
                        else:
                            for ows in range(0,np.shape(water_unique_site_coords)[0]):
                            
                                Ow_site_coord_red = np.transpose(lattice_inv.dot(np.transpose(Ow_site_coord)))
                            
                                for p in range(0,3):
                                    Ow_site_coord_red[0,p] = Ow_site_coord_red[0,p] - np.floor(Ow_site_coord_red[0,p])
                                                    
                                Ow_site_coord = np.transpose(np.transpose(lattice).dot(np.transpose(Ow_site_coord_red)))                       
                            
                                if np.linalg.norm(water_unique_site_coords[ows,:]-Ow_site_coord) < cutoff:
                                    check_prev_site = 1
                                            
                            if check_prev_site == 0:
                                water_unique_site_coords = np.append(water_unique_site_coords, Ow_site_coord, axis=0)
                                n_unique_site += 1
                                
                                H1_coord = -Hbond_dir*OHw_len + Ow_site_coord

                                H2_coord = get_Hcoord(Ow_site_coord, Hbond_dir, lattice, OHw_len)

                                H2_coord_check = 10
                                H2_angle_check = 10

                                try_count_2 = 0

                                while H2_coord_check != 0 or H2_angle_check != 0:

                                    if try_count_2 > max_orient_try:
                                        break

                                    H2_coord = get_Hcoord(Ow_site_coord, Hbond_dir, lattice, OHw_len)

                                    Hbond_dir, H2_coord_check = Hcoord_check(H2_coord, Ow_site_coord, Hbond_dir, coords, lattice, rmin)

                                    H2_angle_check = Hangle_check(H2_coord, Ow_site_coord, H1_coord)

                                    try_count_2 += 1


                                print("\t" + str(1))
                                print("\t" + "\t" + str(try_count_2))

                                water_unique_site_coords_full = np.append(water_unique_site_coords_full, np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1), axis=0)
                                
                                symm_coords = get_symmetry(Ow_site_coord, H1_coord, H2_coord, lattice)
                        
                                water_unique_site_coords_full = np.append(water_unique_site_coords_full, symm_coords, axis=0)
                        
                                water_unique_site_coords = np.append(water_unique_site_coords, symm_coords[:,0:3], axis=0)

                                gen_POSCAR_dec = int(input("Do you want to generate input file (1 for Y and 0 for N): "))

                                if gen_POSCAR_dec == 1:

                                    os.system("touch struc_log.txt")
                                    os.system("echo 'First number is n_unique_site number, second line is the try_count, third number is try_count for second H' >> struc_log.txt")
                                    os.system("echo " + str(n_unique_site) + " >> struc_log.txt")
                                    os.system("echo 'H1 fixed' >> struc_log.txt")
                                    os.system("echo " + str(try_count_2) + " >> struc_log.txt")

                                    if try_count_2 > max_orient_try:
                                        os.system("echo 'WARNING' >> struc_log.txt")

                                    os.system("echo '****' >> struc_log.txt")

                                    gen_POSCAR(np.append(np.append(Ow_site_coord, np.append(H1_coord, H2_coord, axis=1), axis=1), symm_coords, axis=0), coords, lattice, atom_label_all, natoms, n_unique_site, struc_i, substruc_i)
                                
        #print(water_unique_site_coords_full)
        print(np.shape(water_unique_site_coords_full))
        print(n_unique_site)
        
        os.chdir("./..")

    os.chdir("./..")


