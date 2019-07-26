from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager, save_object, load_object
from pyfeti.src.cyclic import Cyclic_Constraint
from pyfeti.src.linalg import Matrix, vector2localdict
from pyfeti.src.feti_solver import SerialFETIsolver, SolverManager
from pyfeti.cases.case_generator import FETIcase_builder
from scipy import sparse
import amfe
import numpy as np
import matplotlib.pyplot as plt
import os, copy
import time
from datetime import datetime



case_id = 512
case_folder = 'case_' +  str(case_id)

def eig2freq(eigval_,Vp):
    new_id = np.argsort(eigval_)[::-1]
    val_p = 1/eigval_.real[new_id]
    freq_p = np.sqrt(val_p)/(2.0*np.pi)
    Vp = Vp[:,new_id]
    return freq_p, Vp

def filepath(filename):
    return os.path.join(case_folder,filename)

def print_date(msg=''):
    print(msg)
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(date_str)

def load_pkl(variable_name):
    return load_object(filepath(variable_name))

def save_pkl(variable, variable_name):
    return save_object(variable, filepath(variable_name))

def update_system(system_list,V):
    v_dict = vector2localdict(V,global2local_primal_dofs)
    for i,sysi in enumerate(system_list):
        vi = v_dict[i+1]
        sysi.u_output = list(vi.T)
    return system_list

def plot_system_list(system_list,mode_id,factor):
    fig1 = plt.figure(figsize=(10,10))
    ax2 = fig1.add_subplot(1,1,1, projection='3d')
    bc = np.array([-200,200])*0.6
    ax2.set_xlim(bc)
    ax2.set_ylim(bc)
    ax2.set_zlim(bc)
    plot_modes(system_list,ax2,mode_id,factor)
    
def plot_modes(system_list,ax2,mode_id,factor):
    for i,sysi in enumerate(system_list):
        amfe.plot_3D_displacement(sysi,displacement_id=(mode_id-1),ax=ax2,factor=factor,plot_nodes=False,scale=1000,alpha=1.0,linewidth=0.05)
        
    
def plot_mode_list(system_list,mode_list,factor=10,freq_list=[]):
    
    figX = plt.figure(figsize=(8,8))
    num_of_modes = len(mode_list)
    
    if num_of_modes<=4:
        
        dx,dy = 2,2
    else:
        dx,dy = 3,num_of_modes//3
    
    count=1
    for mode_id in mode_list: 
        ax = figX.add_subplot(dx, dy, count, projection='3d')
        plot_modes(system_list,ax,mode_id,factor)
        ax.set_xlabel('x-axis [mm]')
        ax.set_ylabel('y-axis [mm]')
        ax.set_zlabel('z-axis [mm]')
        bc = np.array([-200,200])*0.6
        ax.set_xlim(bc)
        ax.set_ylim(bc)
        ax.set_zlim(bc)
        try:
            freq = freq_list[mode_id-1]
            ax.set_title('Mode %i, Frequency = %2.2f [Hz] ' %(mode_id,freq),fontsize=12 )
        except:
            ax.set_title('Mode %i, ' %(mode_id),fontsize=12 )
        count+=1
    return figX

def run_case(tol_factor = 6):
    Nsectors = 24
    domain_label = 4
    cyclic_left_label = 3
    cyclic_right_label = 2
    dirichlet_label = 1
    unit='deg'
    tol_radius = 1.0e-5
    dimension=3
    FETI_tolerance = 10.0**-tol_factor


    feti_obj = SerialFETIsolver(K_dict,B_dict,f_dict,tolerance=FETI_tolerance,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
    manager = feti_obj.manager 
    manager.build_local_to_global_mapping()
    scaling = manager.assemble_global_scaling()
    S = sparse.diags(1./scaling)
    B = manager.assemble_global_B()
    B_ = B 
    P = sparse.eye(K.shape[0]) - S.dot(B_.T.dot(B_))

    print_date('Loading Matrix')
  
    def system_without_projection(u,tol=1.0e-8):
        f = M.dot(u)
        f_dict = manager.vector2localdict(f,manager.global2local_primal_dofs)
        feti_obj = SerialFETIsolver(K_dict,B_dict,f_dict,tolerance=tol,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
        solution_obj = feti_obj.solve()
        u_dict = solution_obj.u_dict
        return solution_obj.displacement
        
    D_wp = sparse.linalg.LinearOperator(shape=M.shape,matvec = lambda x : system_without_projection(x,tol=FETI_tolerance))

    nmodes = 30
    np.random.seed(1)
    u0 = np.random.rand(D_wp.shape[0])
    u0 /= np.linalg.norm(u0)


    print_date('Solving Classical Dual Eigenproblem')
    eigval_without_projection_, V_wp_ = sparse.linalg.eigsh(D_wp,k=nmodes,v0=u0)

    freq_dual_wp_, V_wp_ = eig2freq(eigval_without_projection_,V_wp_)
    save_pkl(V_wp_,'%i_cyclic_V_wp_.pkl' %tol_factor)
    save_pkl(freq_dual_wp_,'%i_cyclic_freq_dual_wp_.pkl' %tol_factor)


    def system(u,tol=1.0e-8):
        f = P.T.dot(M.dot(P.dot(u)))
        f_dict = manager.vector2localdict(f,manager.global2local_primal_dofs)
        feti_obj = SerialFETIsolver(K_dict,B_dict,f_dict,tolerance=tol,pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-8})
        solution_obj = feti_obj.solve()
        u_dict = solution_obj.u_dict
        return P.dot(solution_obj.displacement)


    D = sparse.linalg.LinearOperator(shape=M.shape,matvec = lambda x : system(x,tol=FETI_tolerance))

    print_date('Solving Projected Dual Eigenproblem')
    eigval, V = sparse.linalg.eigsh(D,k=nmodes,v0 = P.dot(u0))

    freq_dual, V = eig2freq(eigval, V)

    save_pkl(V,'%i_cyclic_V.pkl' %tol_factor)
    save_pkl(freq_dual,'%i_cyclic_freq_dual.pkl' %tol_factor)


    freq_list = [freq_dual,freq_dual_wp_]
    V_list = [V,V_wp_]
    return freq_list, V_list



system_list = load_object('case_512/system_list.pkl') 
global2local_primal_dofs = load_object('case_512/global2local_primal_dofs.pkl')
#B = load_pkl('B.pkl')
K = load_pkl('K.pkl')
M = load_pkl('M.pkl')
L = load_pkl('L.pkl')
Lexp = load_pkl('Lexp.pkl')
K_dict = load_pkl('K_dict.pkl')
M_dict = load_pkl('M_dict.pkl')
B_dict = load_pkl('B_dict.pkl')
f_dict = load_pkl('f_dict.pkl')

B_new_dict = B_dict.copy()
for domain_id, B_local in B_dict.items():
    for pair, Bij in B_local.items():
        local_id,nei_id = pair
        if local_id>nei_id and local_id<24:
            B_new_dict[domain_id][pair] = -1.0*Bij
        elif domain_id==24 and nei_id==1:
            continue
        else:
            continue


B_dict = B_new_dict
tol_list = [6,8,9]
for tol_factor in tol_list:
    freq_list, V_list = run_case(tol_factor)
    V,V_wp_ = V_list
    freq_dual,freq_dual_wp_ = freq_list


    mode_list = [1,4,6,10]

    system_list = update_system(system_list,V)
    fig1 = plot_mode_list(system_list,mode_list,factor=1,freq_list=freq_dual)
    fig1.savefig('case_512/%i_dual_cylic_mode_shapes_4by4.png' %tol_factor, dpi=1000)

    system_list = update_system(system_list,V_wp_)
    fig2 = plot_mode_list(system_list,mode_list,factor=1,freq_list=freq_dual_wp_)
    fig2.savefig('case_512/%i_dual_wp__cylic_mode_shapes4by4.png' %tol_factor, dpi=1000)



