from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager, save_object, load_object
from pyfeti.src.cyclic import Cyclic_Constraint
from pyfeti.src.linalg import Matrix
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

def print_date(msg=''):
    print(msg)
    date_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    print(date_str)

def filepath(filename):
    return os.path.join(case_folder,filename)

print_date('Begining')


Nsectors = 24
domain_label = 4
cyclic_left_label = 3
cyclic_right_label = 2
dirichlet_label = 1
unit='deg'
tol_radius = 1.0e-5
dimension=3


try:
    os.mkdir(case_folder)
except:
    pass

print_date('Load Matrix')

def load_pkl(variable_name):
    return load_object(filepath(variable_name))

def save_pkl(variable, variable_name):
    return save_object(variable, filepath(variable_name))

B = load_pkl('B.pkl')
M = load_pkl('M.pkl')
K = load_pkl('K.pkl')
L = load_pkl('L.pkl')
Lexp = load_pkl('Lexp.pkl')


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

save_pkl(Vp,'Vp.pkl')
save_pkl(eigval_,'eigval_.pkl')

print_date('Solving Eigenvalue')

Vprimal = Lexp.dot(Vp)
save_pkl(Vprimal,'Vprimal.pkl')


print_date('End')

