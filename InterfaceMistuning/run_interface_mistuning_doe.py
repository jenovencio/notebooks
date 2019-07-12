#!/usr/bin/env python
# coding: utf-8


from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DiskCreator,  save_object, load_object  
from pyfeti.src.utils import DofManager, OrderedDict, SelectionOperator
from pyfeti.src.feti_solver import SerialFETIsolver, SolverManager, cyclic_eig
from pyfeti.src.linalg import ProjLinearSys
from pyfeti.src.cyclic import Cyclic_Constraint, Contact, Cyclic_Contact
from pyfeti.cases.case_generator import FETIcase_builder
from scipy import sparse
from scipy.sparse import linalg
import amfe
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def translator(string): 
    
    if string[0] == 'F':
        return int(string.split('_')[0][3:])
    elif string[0] == 'S':
        return int('99'.join(string.split('_')[2:4]))
    else:
        return 0


disk_mesh_file = 'disk_sector'
folder = 'meshes'

m2 = load_object(os.path.join(folder, disk_mesh_file + '.pkl'),tries=1)
if m2 is None:
    sector_mesh_file = os.path.join(folder, disk_mesh_file + '.inp')
    m2 = amfe.Mesh()
    m2.import_inp(sector_mesh_file)

    count=1
    for name in m2.get_phys_group_types():
        m2.change_tag_in_eldf('phys_group',name,translator(name))
        count+=1
    save_object(m2,os.path.join(folder, disk_mesh_file + '.pkl'))


blade_mesh_file = 'blade_sector'
m3 = load_object(os.path.join(folder, blade_mesh_file + '.pkl'),tries=1)
if m3 is None:
    sector_mesh_file = os.path.join(folder, blade_mesh_file + '.inp')
    m3 = amfe.Mesh()
    m3.import_inp(sector_mesh_file)

    count=1
    for name in m3.get_phys_group_types():
        m3.change_tag_in_eldf('phys_group',name,translator(name))
        count+=1

    save_object(m3,os.path.join(folder, blade_mesh_file + '.pkl'))




if False:
    fig, ax1 = plt.subplots(1,1,figsize=(10,10))
    amfe.plotmesh(m2,ax=ax1,color='grey')

    mult=1.2
    for ax in (ax1,):
        ax.set_aspect('equal')
        ax.set_xlabel('Width [m]')
        ax.set_ylabel('Heigh [m]')
    plt.legend('off')


disk_mesh_list = [m2.rot_z(i*360/5) for i in range(5)]
blade_mesh_list = [m3.rot_z(i*360/5) for i in range(5)]

def plot3():
    fig, ax2 = plt.subplots(1,1,figsize=(8,8))
    for m in disk_mesh_list:
        amfe.plotmesh(m,ax=ax2,color='grey')
    ax2.set_aspect('equal')
    ax2.set_xlabel('Width [m]')
    ax2.set_ylabel('Heigh [m]')
    plt.legend('off')

def plot4():
    fig, ax1 = plt.subplots(1,1,figsize=(8,8))
    amfe.plotmesh(m3,ax=ax1,color='grey')
    mult=1.2
    for ax in (ax1,):
        ax.set_aspect('equal')
        ax.set_xlabel('Width [m]')
        ax.set_ylabel('Heigh [m]')
    plt.legend('off')

def plot5():
    fig, ax3 = plt.subplots(1,1,figsize=(8,8))
    for m in blade_mesh_list:
        amfe.plotmesh(m,ax=ax3,color='grey')
    ax3.set_aspect('equal')
    ax3.set_xlabel('Width [m]')
    ax3.set_ylabel('Heigh [m]')
    plt.legend('off')


# creating material
my_material = amfe.KirchhoffMaterial(E=210.0E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=1.0)

K_dict = {}
M_dict = {}
count = 1
for m in disk_mesh_list:
    my_system1 = amfe.MechanicalSystem()
    my_system1.set_mesh_obj(m)
    my_system1.set_domain(1,my_material)
    K1, _ = my_system1.assembly_class.assemble_k_and_f()
    M1 = my_system1.assembly_class.assemble_m()
    K_dict[count] = K1
    M_dict[count] = M1
    count+=1
    
for m in blade_mesh_list:
    my_system1 = amfe.MechanicalSystem()
    my_system1.set_mesh_obj(m)
    my_system1.set_domain(11,my_material)
    K1, _ = my_system1.assembly_class.assemble_k_and_f()
    M1 = my_system1.assembly_class.assemble_m()
    K_dict[count] = K1
    M_dict[count] = M1
    count+=1
    

try:
    connectivity = []
    for _,item in m2.el_df.iloc[:, m2.node_idx:].iterrows():
        connectivity.append(list(item.dropna().astype(dtype='int64')))
    m2.el_df['connectivity'] = connectivity
except:
    pass

nnodes = len(m2.nodes)
id_matrix = np.array(range(nnodes*2)).reshape(nnodes,2)

id_dict = {}
for i in range(len(id_matrix)):
    id_dict[i] = list(id_matrix[i])
    
id_map_df = dict2dfmap(id_dict )
    
el_df = m2.el_df


try:
    connectivity = []
    for _,item in m3.el_df.iloc[:, m3.node_idx:].iterrows():
        connectivity.append(list(item.dropna().astype(dtype='int64')))
    m3.el_df['connectivity'] = connectivity
except:
    pass

nnodes = len(m3.nodes)
id_matrix = np.array(range(nnodes*2)).reshape(nnodes,2)

id_dict3 = {}
for i in range(len(id_matrix)):
    id_dict3[i] = list(id_matrix[i])
    
id_map_df3 = dict2dfmap(id_dict3 )
    
el_df3 = m3.el_df


dof_manager = DofManager(el_df,id_map_df)
dof_manager3 = DofManager(el_df3,id_map_df3)

master_nodes = dof_manager.get_node_list_from_group_id(102)
slave_nodes = dof_manager.get_node_list_from_group_id(105)
dir_nodes = dof_manager.get_node_list_from_group_id(101)
master_nodes = list(set(master_nodes).difference(dir_nodes))

cyc_obj = Cyclic_Contact(master_nodes,slave_nodes,m2.nodes,sector_angle=360/5,dimension=2)
pair_dict = cyc_obj.find_node_pairs()
left_nodes = cyc_obj.master_nodes
right_nodes = cyc_obj.slave_nodes


interface_node_1 = dof_manager.get_node_list_from_group_id(111)
interface_node_2 = dof_manager3.get_node_list_from_group_id(1101)
contact_obj = Contact(interface_node_1,interface_node_2,m2.nodes,nodes_coord_slave=m3.nodes)
contact_pair = contact_obj.find_node_pairs(tol_radius=4.0E-02)


d = {}
d[1,2] = dof_manager.get_dofs_from_node_list(left_nodes,direction='xy')
d[1,5] = dof_manager.get_dofs_from_node_list(right_nodes,direction='xy')
d[1,6] = dof_manager.get_dofs_from_node_list(interface_node_1,direction='xy')
d[1,1] = dof_manager.get_dofs_from_node_list(dir_nodes ,direction='xy')

d2 = {}
d2[6,1] = dof_manager3.get_dofs_from_node_list(interface_node_2,direction='xy')


s = SelectionOperator(d,id_map_df)
s2 = SelectionOperator(d2,id_map_df3)

def case1():
    dict_key = (1,5)
    K1 = K_dict[1]
    M1 = M_dict[1]
    B = s.build_B(dict_key).A
    BBT_inv = np.linalg.inv(B.dot(B.T))
    P = np.eye(B.shape[1]) - B.T.dot(BBT_inv.dot(B))

    obj = ProjLinearSys(K1.A,M1.A,P)
    Dp = obj.getLinearOperator()

    v0 = np.random.rand(K1.shape[0])
    nmodes= 9
    eigval_, Vp = sparse.linalg.eigsh(Dp,k=nmodes,v0=P.dot(v0))

    val_wp_ = np.sort(1/eigval_)
    freq_wp_ = np.sqrt(val_wp_)/(2.0*np.pi)
    freq_wp_


    # creating material
    my_material = amfe.KirchhoffMaterial(E=210.0E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=1.0)

    my_system1 = amfe.MechanicalSystem()
    my_system1.set_mesh_obj(m2)
    my_system1.set_domain(1,my_material)

    m = 1
    my_system1.u_output = list(Vp.T)


def plot6(f):
    fig1, ax1_list = plt.subplots(3,3,figsize=(10,5))
    counter = 0
    delta_ = 1.0
    for ax_ij in ax1_list:
        for ax2 in ax_ij:
            amfe.plot_2D_system_solution(my_system1,u_id=(nmodes - 1 - counter),ax=ax2,factor=f)
            ax2.set_aspect('equal')
            ax2.set_xlabel('Width [m]')
            ax2.set_ylabel('Heigh [m]')
            ax2.set_title('Mode id = %i' %(counter+1) )
            counter+=1
    plt.legend('off')
    


def map_neighbors(domain_key):
    pair = []
    delta = [(0,0),(0,1),(0,-1),(0,5)]
    
    if domain_key==5:
        delta[1] = (0,-1) 
    elif domain_key==1:
        delta[2] = (0,4)
    else:
        pass
        
    for i,j in delta:
        ip,jp = domain_key+i,domain_key+j
        pair.append((ip,jp))
        
    return pair
            

B_dict = {}
for domain_id in K_dict: 
    B_local = {}
    if domain_id<6:
        for interface_pairs, boolean_index in zip(map_neighbors(domain_id),map_neighbors(1)):
            B_local[interface_pairs] = s.build_B(boolean_index)
    else:
        B_local[(domain_id,domain_id-5)] = -1.0*s2.build_B((6,1))
        
    B_dict[domain_id] = B_local
     

def case2():
    Ks_dict = {}
    Ks_dict[1] = K_dict[1]
    Ks_dict[2] = K_dict[6]

    Ms_dict = {}
    Ms_dict[1] = M_dict[1]
    Ms_dict[2] = M_dict[6]

    Bs_dict = {}
    Bs_dict[1] = {(1,1) : B_dict[1][1,1], (1,2) : B_dict[1][1,6] }
    Bs_dict[2] = {(2,1) : B_dict[6][6,1]}


    fs_dict = {1 : np.zeros(K_dict[1].shape[0]),
            2 : np.zeros(K_dict[6].shape[0])} 

    K_feti_obj = SerialFETIsolver(Ks_dict,Bs_dict,fs_dict)
    M_feti_obj = SerialFETIsolver(Ms_dict,Bs_dict,fs_dict)

    Ks, _ = K_feti_obj.manager.assemble_global_K_and_f()
    Ms, _ = M_feti_obj.manager.assemble_global_K_and_f()
    B = K_feti_obj.manager.assemble_global_B()
    from scipy.sparse import linalg 
    BBT_inv = linalg.inv(B.dot(B.T))
    Ps = np.eye(B.shape[1]) - B.T.dot(BBT_inv.dot(B))


    obj = ProjLinearSys(Ks.A,Ms.A,Ps.A)
    Dp = obj.getLinearOperator()

    v0 = np.random.rand(Ks.shape[0])
    nmodes= 9
    eigval_, Vp = sparse.linalg.eigsh(Dp,k=nmodes,v0=Ps.A.dot(v0))

    val_wp_ = np.sort(1/eigval_)
    freq_wp_ = np.sqrt(val_wp_)/(2.0*np.pi)
    freq_wp_

    ansys_freq = np.array([4.5495,39.845,59.994,72.253,131.34,148.82,169.8,197.68,231.13])


def plot_sector_solution():

    # creating material
    my_material = amfe.KirchhoffMaterial(E=210.0E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=1.0)

    my_system1 = amfe.MechanicalSystem()
    my_system1.set_mesh_obj(m2)
    my_system1.set_domain(1,my_material)

    my_system2 = amfe.MechanicalSystem()
    my_system2.set_mesh_obj(m3)
    my_system2.set_domain(11,my_material)

    manager = K_feti_obj.manager
    v_dict = manager.vector2localdict(Vp,manager.global2local_primal_dofs)
    p0 = 10.0
    u1=p0*v_dict[1]
    u2=p0*v_dict[2]
    my_system1.u_output = list(u1.T)
    my_system2.u_output = list(u2.T)

    fig, ax1_list = plt.subplots(3,3,figsize=(10,10))
    counter = 0
    delta_ = 1.0
    for ax_ij in ax1_list:
        for ax2 in ax_ij:
            amfe.plot_2D_system_solution(my_system1,u_id=(counter),ax=ax2)
            amfe.plot_2D_system_solution(my_system2,u_id=(counter),ax=ax2)
            ax2.set_aspect('equal')
            ax2.set_xlabel('Width [m]')
            ax2.set_ylabel('Heigh [m]')
            ax2.set_title('Mode id = %i' %(counter+1) )
            counter+=1
    plt.legend('off')
    plt.tight_layout()



f_dict = {}
for i, K in K_dict.items():
    f_dict[i] = np.zeros(K.shape[0]) 

K_feti_obj = SerialFETIsolver(K_dict,B_dict,f_dict)
M_feti_obj = SerialFETIsolver(M_dict,B_dict,f_dict)

K, _ = K_feti_obj.manager.assemble_global_K_and_f()
M, _ = M_feti_obj.manager.assemble_global_K_and_f()
B = K_feti_obj.manager.assemble_global_B()
from scipy.sparse import linalg 
BBT_inv = linalg.inv(B.dot(B.T))
P = sparse.eye(B.shape[1]) - B.T.dot(BBT_inv.dot(B))

#Kinv = np.linalg.pinv(K.A)
#precond = linalg.LinearOperator(shape=K.shape, matvec = lambda x : P.dot(Kinv.dot(P.dot(x))))
precond = None
obj = ProjLinearSys(K,M,P,precond=precond,dtype=np.float)
Dp = obj.getLinearOperator()

v0 = np.random.rand(K.shape[0])

print('Standard Case')
start = time.time()
nmodes= 10
eigval_, Vp = sparse.linalg.eigsh(Dp,k=nmodes,v0=P.dot(v0))
print('Elaspsed time = %2.2f ' %(time.time() - start) )

val_wp_ = np.sort(1/eigval_)
freq_wp_ = np.sqrt(val_wp_)/(2.0*np.pi)

print('Computed Eigen frequencies [Hz]', freq_wp_)
ansys_freq = np.array([4.5635, 4.5738, 4.5741, 4.5774, 4.5775, 47.546, 53.643, 53.644, 54.672])
print('Ansys Eigen frequencies [Hz]', ansys_freq)

# creating material
my_material = amfe.KirchhoffMaterial(E=210.0E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=1.0)

sys_dict = {}
count=1
for m in disk_mesh_list:
    sys = amfe.MechanicalSystem()
    sys.set_mesh_obj(m)
    sys.set_domain(1,my_material)
    sys_dict[count] = sys
    count+=1
    
for m in blade_mesh_list:
    sys = amfe.MechanicalSystem()
    sys.set_mesh_obj(m)
    sys.set_domain(11,my_material)
    sys_dict[count] = sys
    count+=1


manager = K_feti_obj.manager
v_dict = manager.vector2localdict(Vp,manager.global2local_primal_dofs)
p0 = 10.0

for i,sys in sys_dict.items():
    u1 = v_dict[i]
    sys.u_output = list(u1.T)

def plot_solution():
    fig, ax1_list = plt.subplots(3,3,figsize=(20,20))
    counter = 0
    delta_ = 1.0
    for ax_ij in ax1_list:
        for ax2 in ax_ij:
            for i,sys in sys_dict.items():
                amfe.plot_2D_system_solution(sys,u_id=(counter),ax=ax2,factor=20)
            
            ax2.set_aspect('equal')
            ax2.set_xlabel('Width [m]')
            ax2.set_ylabel('Heigh [m]')
            ax2.set_title('Mode id = %i' %(counter+1) )
            counter+=1
    plt.legend('off')
    plt.tight_layout()



def solve_eig(K_dict,M_dict,B_dict,nmodes=10,precond=None):
    f_dict = {}
    for i, K in K_dict.items():
        f_dict[i] = np.zeros(K.shape[0]) 

    K_feti_obj = SerialFETIsolver(K_dict,B_dict,f_dict)
    M_feti_obj = SerialFETIsolver(M_dict,B_dict,f_dict)

    K, _ = K_feti_obj.manager.assemble_global_K_and_f()
    M, _ = M_feti_obj.manager.assemble_global_K_and_f()
    B = K_feti_obj.manager.assemble_global_B()
    from scipy.sparse import linalg 
    BBT_inv = linalg.inv(B.dot(B.T))
    P = sparse.eye(B.shape[1]) - B.T.dot(BBT_inv.dot(B))

    obj = ProjLinearSys(K,M,P,precond=precond)
    Dp = obj.getLinearOperator()

    v0 = np.random.rand(K.shape[0])
    eigval_, Vp = sparse.linalg.eigsh(Dp,k=nmodes,v0=P.dot(v0))
    print(eigval_)
    val_wp_ = np.sort(1/eigval_)
    freq_wp_ = np.sqrt(val_wp_)/(2.0*np.pi)
    return freq_wp_, Vp, B.shape[0]

#------------------------------------------------------------------------------
#  Start Statistical Study
#-----------------------------------------------------------------------------
np.random.seed(1)
nsamples = 20
ninterfaces = 5
nc = B_dict[1][1,6].shape[0]
doe = np.random.randint(2, size=(nsamples,ninterfaces,nc))

save_object(doe,'doe.pkl')

import copy
interface_pair_list = [(1,6),(2,7),(3,8),(4,9),(5,10)]
Bp_list = []
for sample in doe:
    B_dict_ = copy.deepcopy(B_dict.copy())
    for interface_pair, pindex in zip(interface_pair_list,sample): 
        domain_i, domain_j = interface_pair
        Bij = copy.deepcopy(B_dict_[domain_i][domain_i, domain_j])
        Bji = copy.deepcopy(B_dict_[domain_j][domain_j, domain_i])
        p = sparse.diags(pindex)
        pos = np.argwhere(pindex>0).T[0]
        B_dict_[domain_i][domain_i, domain_j] = Bij[pos,:]
        B_dict_[domain_j][domain_j, domain_i] = Bji[pos,:]
    Bp_list.append(B_dict_)

print('Running Interface Mistuning Perturbation')
freq_list = []
V_list = []
nint_list = []
nmodes=10
count=0
for B_dict_p in Bp_list:
    time_start = time.time()
    print('Running sample %i' %count)
    freq, Vp, nint = solve_eig(K_dict,M_dict,B_dict_p,nmodes=nmodes,precond=precond)
    print('Elapsed time = %2.4f' %(time.time() - time_start))
    print('Computed Eigen Frequencies [Hz]', freq)
    freq_list.append(freq)
    V_list.append(Vp)
    nint_list.append(nint)
    count+=1

save_object(V_list,'eigenshape_list.pkl')
save_object(nint_list,'nint_list.pkl')
save_object(freq_list,'freq_list.pkl')

