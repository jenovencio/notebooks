from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager, save_object
from pyfeti.src.feti_solver import SerialFETIsolver, SolverManager
from pyfeti.cases.case_generator import FETIcase_builder
from scipy import sparse
import amfe
import numpy as np
import matplotlib.pyplot as plt
import copy

def save_pictute(m1,fig_file):
    fig1 = plt.figure(figsize=(10,10))
    ax1 = fig1.add_subplot(1,1,1, projection='3d')
    amfe.plot3Dmesh(m1,ax=ax1,scale=1000, plot_nodes=False,boundaries=False, alpha=1.0,linewidth=0.05)
    bc = np.array([-200,200])
    ax1.set_xlim(bc)
    ax1.set_ylim(bc)
    ax1.set_zlim(bc)
    ax1.set_xlabel('x-axis [mm]')
    ax1.set_ylabel('y-axis [mm]')
    ax1.set_zlabel('z-axis [mm]')
    ax1.grid(False)
    fig1.savefig(fig_file,dpi=1000)


Nsectors = 24
domain_label = 4
cyclic_left_label = 3
cyclic_right_label = 2
dirichlet_label = 1
unit='deg'
tol_radius = 1.0e-5
dimension=3

case_list = [512,1524,2649,16480]

for case_id in case_list:
    mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_' + str(case_id) + '_nodes.inp'
    pkl_file = 'meshes/3D_simple_bladed_disk_24_sectors_' + str(case_id) + '_nodes.pkl'
    fig_file = 'meshes/3D_simple_bladed_disk_24_sectors_' + str(case_id) + '_nodes.png'

    scale = 1.0
    m1 = amfe.Mesh()
    m1.import_inp(mesh_file,scale)


    m1.change_tag_in_eldf('phys_group','RIGHT_ELSET',cyclic_right_label)
    m1.change_tag_in_eldf('phys_group','LEFT_ELSET',cyclic_left_label )
    m1.change_tag_in_eldf('phys_group','BODY_1_1_SOLID_ELSET',domain_label)
    m1.change_tag_in_eldf('phys_group','BODY_1_1_ELSET',5)
    m1.change_tag_in_eldf('phys_group','DIRICHLET_ELSET',dirichlet_label)

    save_object(m1,pkl_file)

    save_pictute(m1,fig_file)








