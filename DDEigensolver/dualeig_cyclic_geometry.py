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

def print_date(msg=''):
    print(msg)
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(date_str)

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
m1 = load_object('3D_simple_bladed_disk_24_sectors_512_nodes.pkl')
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


m1.change_tag_in_eldf('phys_group','RIGHT_ELSET',cyclic_right_label)
m1.change_tag_in_eldf('phys_group','LEFT_ELSET',cyclic_left_label )
m1.change_tag_in_eldf('phys_group','BODY_1_1_SOLID_ELSET',domain_label)
m1.change_tag_in_eldf('phys_group','BODY_1_1_ELSET',5)
m1.change_tag_in_eldf('phys_group','DIRICHLET_ELSET',dirichlet_label)


# creating material
my_material = amfe.KirchhoffMaterial(E=210.0E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=1.0)

my_system1 = amfe.MechanicalSystem()
my_system1.set_mesh_obj(m1)
my_system1.set_domain(4,my_material)

K1, _ = my_system1.assembly_class.assemble_k_and_f()
M1 = my_system1.assembly_class.assemble_m()

el_df = copy.deepcopy(m1.el_df)
try:
    connectivity = []
    for _,item in el_df.iloc[:, m1.node_idx:].iterrows():
        connectivity.append(list(item.dropna().astype(dtype='int64')))
    el_df['connectivity'] = connectivity
except:
    pass
    


sector_angle = 360/Nsectors 
id_matrix = my_system1.assembly_class.id_matrix
id_map_df = dict2dfmap(id_matrix)
nodes_coord = m1.nodes

cyc_obj = Cyclic_Constraint(id_map_df,
                            el_df,
                            nodes_coord,
                            dirichlet_label,
                            cyclic_left_label,
                            cyclic_right_label,
                            sector_angle,
                            unit=unit,
                            tol_radius = tol_radius,
                            dimension=dimension)

translate_dict = {}
translate_dict['d'] = dirichlet_label
translate_dict['r'] = cyclic_right_label
translate_dict['l'] = cyclic_left_label

s = cyc_obj.s
B_local_dict = {}
for key, value in translate_dict.items():
    B_local_dict[value] = s.build_B(key)


mesh_list = [m1.rot_z(i*360/Nsectors) for i in range(Nsectors)]
#plot_mesh_list(mesh_list)

system_list = []
K_dict = {}
M_dict = {}
B_dict = {}
f_dict = {}
for i,mi in enumerate(mesh_list):
    sysi = amfe.MechanicalSystem()
    sysi.set_mesh_obj(mi)
    sysi.set_domain(domain_label,my_material)
    system_list.append(sysi)
    K1, _ = sysi.assembly_class.assemble_k_and_f()
    M1 = sysi.assembly_class.assemble_m()
    K_dict[i+1] = Matrix(K1,key_dict=s.selection_dict).eliminate_by_identity('d')
    M_dict[i+1] = Matrix(M1,key_dict=s.selection_dict).eliminate_by_identity('d',multiplier=0.0)
    plus = +1
    minus = -1
    local_index = i+1
    if i+2>Nsectors:
        plus = -23

    if i-1<0:
        minus = +23

    sign_plus = np.sign(plus)
    sign_minus = np.sign(plus)

    B_dict[local_index] = {(local_index,local_index+plus): sign_plus*B_local_dict[cyclic_left_label],
                            (local_index,local_index+minus): sign_minus*B_local_dict[cyclic_right_label]}

    f_dict[local_index] = np.zeros(K1.shape[0])

feti_obj1 = SerialFETIsolver(K_dict,B_dict,f_dict,tolerance=1.0e-12,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
feti_obj2 = SerialFETIsolver(M_dict,B_dict,f_dict,tolerance=1.0e-12,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
manager = feti_obj1.manager 
managerM = feti_obj2.manager
manager.build_local_to_global_mapping()

print('Loading Matrix')
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(date_str)


B = load_object('B.pkl')
M = load_object('M.pkl')
K = load_object('K.pkl')
L = load_object('L.pkl')
Lexp = load_object('Lexp.pkl')


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


BBT_inv_lu = sparse.linalg.splu(B.dot(B.T))

#scaling = manager.assemble_global_scaling()
#S = np.diag(1./scaling)
#BBT_inv_tilde = B.dot(S).dot(S.dot(B.T))
BBT_inv_tilde = sparse.linalg.LinearOperator(shape=(B.shape[0],B.shape[0]), matvec = lambda x : BBT_inv_lu.solve(x) )
P = sparse.linalg.LinearOperator(shape=K.shape, matvec = lambda x : x - B.T.dot(BBT_inv_tilde.dot(B.dot(x))))


countswp=0
def system_without_projection(u,tol=1.0e-8):
    
    global countswp
    f = M.dot(u)
    f_dict = manager.vector2localdict(f,manager.global2local_primal_dofs)
    feti_obj = SerialFETIsolver(K_dict,B_dict,f_dict,tolerance=tol,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
    solution_obj = feti_obj.solve()
    u_dict = solution_obj.u_dict
    countswp+=1
    return solution_obj.displacement
    
D_wp = sparse.linalg.LinearOperator(shape=M.shape,matvec = lambda x : system_without_projection(x))

nmodes = 30
np.random.seed(1)
u0 = np.random.rand(D_wp.shape[0])
u0 /= np.linalg.norm(u0)


print_date('Solving Classical Dual Eigenproblem')
eigval_without_projection_, V_wp_ = sparse.linalg.eigs(D_wp,k=nmodes,v0=u0,maxiter=200)
val_wp_ = np.sort(1/eigval_without_projection_.real)
freq_dual_wp_ = np.sqrt(val_wp_)/(2.0*np.pi)
freq_dual_wp_

save_object(V_wp_,'V_wp_.pkl')
save_object(eigval_without_projection_,'eigval_without_projection_.pkl')


counts=0
def system(u,tol=1.0e-8):
    global counts
    f = P.T.dot(M.dot(P.dot(u)))
    f_dict = manager.vector2localdict(f,manager.global2local_primal_dofs)
    feti_obj = SerialFETIsolver(K_dict,B_dict,f_dict,tolerance=tol,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
    solution_obj = feti_obj.solve()
    u_dict = solution_obj.u_dict
    counts+=1
    return solution_obj.displacement


D = sparse.linalg.LinearOperator(shape=M.shape,matvec = lambda x : system(x))

print_date('Solving Projected Dual Eigenproblem')
eigval, V = sparse.linalg.eigs(D,k=nmodes,v0 = P.dot(u0))
val = np.sort(1/eigval.real)
freq_dual = np.sqrt(val)/(2.0*np.pi)
freq_dual

save_object(V,'V.pkl')
save_object(eigval,'eigval.pkl')

print('End')
date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(date_str)