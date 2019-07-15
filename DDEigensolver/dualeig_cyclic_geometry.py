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



case_id = 512
case_folder = 'case_' +  str(case_id)

def filepath(filename):
    return os.path.join(case_folder,filename)

def print_date(msg=''):
    print(msg)
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(date_str)

def load_pkl(variable_name):
    return load_object(filepath(variable_name))

def save_pkl(variable, variable_name):
    return sabe_object(variable, filepath(variable_name))

Nsectors = 24
domain_label = 4
cyclic_left_label = 3
cyclic_right_label = 2
dirichlet_label = 1
unit='deg'
tol_radius = 1.0e-5
dimension=3


feti_obj1 = load_pkl('feti_obj1.pkl')
feti_obj2 = load_pkl('feti_obj2.pkl')
manager =  load_pkl('manager.pkl')
managerM = load_pkl('managerM.pkl')

B = load_pkl('B.pkl')
K = load_pkl('K.pkl')
M = load_pkl('M.pkl')

print_date('Loading Matrix')
BBT_inv_lu = sparse.linalg.splu(B.dot(B.T))
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

print_date('End')
