from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager, save_object, load_object
from pyfeti.src.cyclic import Cyclic_Constraint
from pyfeti.src.linalg import Matrix
from pyfeti.src.feti_solver import SerialFETIsolver, SolverManager
from pyfeti.cases.case_generator import FETIcase_builder
from scipy import sparse
import amfe
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
from datetime import datetime

def print_date():
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(date_str)

print_date()

Nsectors = 24
domain_label = 4
cyclic_left_label = 3
cyclic_right_label = 2
dirichlet_label = 1
unit='deg'
tol_radius = 1.0e-5
dimension=3

mesh_file = 'meshes/3d_simple_bladed_disk_24_sector.msh'
mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_course_v3.inp'
mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_1524_nodes.inp'
mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_2649_nodes.inp'
#mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_16480_nodes.inp'
#m1 = load_object('3D_simple_bladed_disk_24_sectors_1524_nodes.pkl')
#m1 = load_object('3D_simple_bladed_disk_24_sectors_512_nodes.pkl')
#m1.import_msh(mesh_file)
#m1.import_inp(mesh_file,scale)
#save_object(m1,'3D_simple_bladed_disk_24_sectors_2649_nodes')


def plot1(m1):
    ax1 = amfe.plot3Dmesh(m1,boundaries=False)
    bc = np.array([-200,200])
    ax1.set_xlim(bc)
    ax1.set_ylim(bc)
    ax1.set_zlim(bc)
    plt.show()


def plot_mesh_list(mesh_list):
    fig1 = plt.figure(figsize=(20,20))
    ax1 = fig1.add_subplot(1,1,1, projection='3d')
    for mi in mesh_list:
        amfe.plot3Dmesh(mi,ax=ax1,boundaries=False)
    plt.show()




print('Load Matrix')
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(date_str)



B = load_object('B.pkl')
M = load_object('M.pkl')
K = load_object('K.pkl')
L = load_object('L.pkl')
Lexp = load_object('Lexp.pkl')


print('Assembling primal operator')
print_date()
Kp = L.dot(K.dot(Lexp))
Mp = L.dot(M.dot(Lexp))
lu = sparse.linalg.splu(Kp.tocsc())
Dp = sparse.linalg.LinearOperator(shape=Kp.shape, matvec = lambda  x : lu.solve(Mp.dot(x)))
nmodes = 30


print('Solving Eigenvalue')
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(date_str)
eigval_, Vp = sparse.linalg.eigsh(Dp,k=nmodes)
val_p = np.sort(1/eigval_)
freq_p = np.sqrt(val_p)/(2.0*np.pi)


print('Salving solution')
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(date_str)

save_object(Vp,'Vp.pkl')
save_object(eigval_,'eigval_.pkl')

print('Solving Eigenvalue')
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(date_str)

Vprimal = Lexp.dot(Vp)
save_object(Vprimal,'Vprimal.pkl')

def update_system(system_list,V):
    v_dict = manager.vector2localdict(V,manager.global2local_primal_dofs)
    for i,sysi in enumerate(system_list):
        vi = v_dict[i+1]
        sysi.u_output = list(vi.T)
    return system_list

def plot_system_list(system_list,mode_id):
    fig1 = plt.figure(figsize=(20,20))
    ax1 = fig1.add_subplot(1,1,1, projection='3d')

    for i,sysi in enumerate(system_list):
        amfe.plot_3D_system_solution(sysi,u_id=(nmodes - 1 - mode_id),ax=ax2,factor=20,linewidth=0.1)
        
    plt.legend('off')
    

#system_list = update_system(system_list,Vprimal)


#BBT_inv = np.linalg.pinv(B.A.dot(B.A.T))
#scaling = manager.assemble_global_scaling()
#S = np.diag(1./scaling)
#BBT_inv_tilde = B.dot(S).dot(S.dot(B.T))
#BBT_inv_tilde = BBT_inv
#P = sparse.LinearOperator(shape=K.shape, matvec = lambda x : x - B.T.dot(BBT_inv_tilde.dot(B.dot(x))))

print('End')
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(date_str)
