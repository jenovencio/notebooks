
from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager
from pyfeti.src.feti_solver import SerialFETIsolver, SolverManager
from pyfeti.cases.case_generator import FETIcase_builder
from scipy import sparse
import amfe
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
#from numba import jit, njit, prange
#from numba.typed import Dict, List

def GMRes(A, b, x0, e, nmax_iter, restart=None):
    r = b - np.asarray(np.dot(A, x0)).reshape(-1)

    x = []
    q = [0] * (nmax_iter)

    x.append(r)

    q[0] = r / np.linalg.norm(r)

    h = np.zeros((nmax_iter + 1, nmax_iter))

    for k in range(nmax_iter):
        y = np.asarray(np.dot(A, q[k])).reshape(-1)

        for j in range(k+1):
            h[j, k] = np.dot(q[j], y)
            y = y - h[j, k] * q[j]
        h[k + 1, k] = np.linalg.norm(y)
        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            q[k + 1] = y / h[k + 1, k]

        b = np.zeros(nmax_iter + 1)
        b[0] = np.linalg.norm(r)

        result = np.linalg.lstsq(h, b)[0]

        x.append(np.dot(np.asarray(q).transpose(), result) + x0)

    return x

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

class ParallelMatrix(sparse.linalg.LinearOperator):

    def __init__(self,K_dict,mapdict,shape=None,dtype=np.float):
        super().__init__(shape=shape,dtype=dtype)
        self.K_dict = K_dict
        self.mapdict = mapdict
        self.n = len(K_dict)

    def array2dict(self,v):
        v_dict = {}
        for key, posid in self.mapdict.items():
            v_dict[key] = v[posid]
        return v_dict 
    
    def dict2array(self,v_dict):
        v = np.zeros(self.shape[0],dtype=self.dtype)
        for key, posid in self.mapdict.items():
            v[posid] = v_dict[key]
        return v

    def _matvec(self,v):
        
        vdict = self.array2dict(v)
        u_dict = numbadot(self.K_dict,vdict,self.n)
        return self.dict2array(u_dict)
            

class LO(sparse.linalg.LinearOperator):
    def __init__(self,matvec,shape=None,dtype=np.float):
        super().__init__(shape=shape,dtype=dtype)
        self._matvec_func = matvec
        self.count = 0

    #@timing
    def _matvec(self,v):
        self.count += 1
        return self._matvec_func(v)

    def _matmat(self,v):
        self.count += 1
        return self._matvec_func(v)

#@jit(parallel=False)
def numbadot(K_dict,vdict,n):
    u_dict = {}
    for i in range(n):
        key = i+1
        u_dict[key] = K_dict[key].dot(vdict[key])
    return u_dict






def eig2freq(eigval_,Vp):
    new_id = np.argsort(eigval_)[::-1]
    val_p = 1/eigval_.real[new_id]
    freq_p = np.sqrt(val_p)/(2.0*np.pi)
    Vp = Vp[:,new_id]
    return freq_p, Vp

#alpha=0.00005,beta=0.0000001
buildZ = lambda w,M,K,alpha=0.0005,beta=0.00001 : -w**2*M + K + 1J*w*(alpha*K + beta*M)
def build_Z_dict(w,M_dict,K_dict):
    Z_dict = {}
    for key, K in K_dict.items():
        Z_dict[key] = buildZ(w,M_dict[key],K)  
    return Z_dict

width = 8.
heigh = 2.
divX=21
divY=7
dobj = DomainCreator(width=width,heigh=heigh,x_divisions=divX,y_divisions=divY)
mesh_file = 'mesh.msh'
dobj.save_gmsh_file(mesh_file)

m1 = amfe.Mesh()
m1.import_msh(mesh_file)


domains_X = 2
domains_Y = 1
n_domains = domains_X*domains_Y 
base = np.array([1,1,1])
facecolor_list = []
mesh_list = []
for my in range(domains_Y):
    for mx in range(domains_X):
        mij = m1.translation(np.array([mx*width,my*heigh]))
        mesh_list.append(mij) 
        facecolor_list.append(0.95*base)
#mesh_list = [m1,m2]

fig1, ax1 = plt.subplots(1,1,figsize=(10,3))

for mi in mesh_list :
    amfe.plot2Dmesh(mi,ax=ax1)

ax1.set_xlim([-2,domains_X*width+2])
ax1.set_ylim([-2,domains_Y*heigh+2])
ax1.set_aspect('equal')
ax1.set_xlabel('Width [m]')
ax1.set_ylabel('Heigh [m]')
plt.legend('off')



# creating material
my_material = amfe.KirchhoffMaterial(E=210.0E9, nu=0.3, rho=7.86E3, plane_stress=True, thickness=1.0)

my_system1 = amfe.MechanicalSystem()
my_system1.set_mesh_obj(m1)
my_system1.set_domain(3,my_material)
K1, _ = my_system1.assembly_class.assemble_k_and_f()
M1 = my_system1.assembly_class.assemble_m()


system_list = []
for mi in mesh_list:
    sysi = amfe.MechanicalSystem()
    sysi.set_mesh_obj(mi)
    sysi.set_domain(3,my_material)
    system_list.append(sysi)
    
try:
    connectivity = []
    for _,item in m1.el_df.iloc[:, m1.node_idx:].iterrows():
        connectivity.append(list(item.dropna().astype(dtype='int64')))
    m1.el_df['connectivity'] = connectivity
except:
    pass
    
id_matrix = my_system1.assembly_class.id_matrix
id_map_df = dict2dfmap(id_matrix)
s = create_selection_operator(id_map_df,m1.el_df)

neighbors_dict = {}
neighbors_dict['right'] = 2
neighbors_dict['left'] = 1
neighbors_dict['top'] = 5
neighbors_dict['bottom'] = 4
neighbors_dict['bottom_left_corner'] = 6
neighbors_dict['bottom_right_corner'] = 7
neighbors_dict['top_left_corner'] = 8
neighbors_dict['top_right_corner'] = 9

B_local_dict = {}
for key, value in neighbors_dict.items():
    B_local_dict[key] = s.build_B(value)


zeros = np.zeros(K1.shape[0])
case = FETIcase_builder(domains_X,domains_Y, K1, zeros, B_local_dict,s,BC_type='G',force_scaling=1.0)
K_dict, B_dict, f_dict = case.build_subdomain_matrices()


fbase = B_local_dict['top_right_corner'].T.dot([0.0,-1.0E8])
f_dict = {}
for i in range(n_domains):
    key = i+1
    if key<n_domains:
        f_dict[key] = 0.0*fbase 
    else:
        f_dict[key] = 1.0*fbase 

M_dict = {}
for i in range(len(mesh_list)):
    M_dict[i+1] = M1
    
#K_dict = {1:K1, 2:K2}
feti_obj1 = SerialFETIsolver(K_dict,B_dict,f_dict,tolerance=1.0e-12)
feti_obj2 = SerialFETIsolver(M_dict,B_dict,f_dict,tolerance=1.0e-12)
#solution = feti_obj1.solve()
manager = feti_obj1.manager 
managerM = feti_obj2.manager
Mi = managerM.local_problem_dict[1].K_local
Mi.key_dict = s.selection_dict
Mi.eliminate_by_identity(1,multiplier=0.0)
for i in range(len(mesh_list)):
    M_dict[i+1] = Mi.data[:,:]


mapdict = manager.local2global_lambda_dofs
def array2dict(v):
    v_dict = {}
    for key, posid in mapdict.items():
        if v.ndim<2:
            v_dict[key] = v[posid]
        else:
            v_dict[key] = v[posid,:]
    return v_dict 

K, f = manager.assemble_global_K_and_f()
M, _ = managerM.assemble_global_K_and_f()
B = manager.assemble_global_B()
BBT_inv = sparse.csc_matrix(np.linalg.pinv(B.dot(B.T).A))
P = sparse.eye(K.shape[0]) - B.T.dot(BBT_inv.dot(B))
L = manager.assemble_global_L()
Lexp = manager.assemble_global_L_exp()
Kp = L@K@Lexp
Mp = L@M@Lexp
lu = sparse.linalg.splu(Kp)

Dp = sparse.linalg.LinearOperator(shape=Kp.shape, matvec = lambda x : lu.solve(Mp.dot(x)))
eigval, V = sparse.linalg.eigsh(Dp,k=10)
freq,V = eig2freq(eigval,V)

f_ = 1.0E10*L.dot(f)
#f_ = V.dot(100*np.ones(V.shape[1]))
#f_ = V[:,5]
f = Lexp.dot(f_)


def simu_action(v):
    d = np.zeros(shape=(manager.lambda_size,manager.num_partitions), dtype=manager.dtype)
    v_dict = array2dict(v)
    count = 0 
    for key, local_problem in manager.local_problem_dict.items():
        s_local = local_problem.apply_schur_complement(v_dict,precond_type='Dirichlet')
        v_id_ = list(s_local.keys())[0]
        i_id, j_id = v_id_
        if j_id<i_id:
            v_id = (j_id, i_id)
            sign = 1.0
        else:
            v_id = v_id_
            sign = 1.0
        d[mapdict[v_id],count] = sign*s_local[v_id_]
        #d[mapdict[v_id],count] = sign*v/2
        count+=1 
    return d


solution = feti_obj1.solve()
lambda_target = solution.interface_lambda

d = manager.assemble_global_d()
e = manager.assemble_e()
F = manager.assemble_global_F()
G = manager.assemble_G()
GGT_inv = sparse.linalg.inv(G@G.T) 
P = sparse.linalg.LinearOperator(shape=F.shape, matvec= lambda x: x - G.T@GGT_inv@G.dot(x))

#solve lambda_im
lambda_im = G.T@GGT_inv.dot(e)
d_ = d - F.dot(lambda_im)
li = 0.0*lambda_im

r = P.dot(d_)
Z = simu_action(r)
W = P@Z

Q_list = []
W_list = []
H_list = []

for i in range(6):
    norm = np.sqrt(r.T.dot(r))

    if norm<1.0E-14:
        break

    Q = np.zeros(W.shape)
    for k in range(manager.num_partitions):
        Q[:,k] = F@W[:,k]
    
    g = W.T.dot(r)

    H = np.linalg.pinv(Q.T@W)
    li = li + W@H.dot(g)
    r = r - P@Q@H.dot(g)

    #x, res, rank, s = np.linalg.lstsq(Q,r)
    #li = li + W.dot(x)
    #r = r - P@Q.dot(x)

    Q_list.append(Q)
    W_list.append(W)
    H_list.append(H)

    Z = simu_action(r)
    W = P@Z
    for j in range(i+1):
        Qj = Q_list[j]
        Wj = W_list[j]
        Hj = H_list[j]
        Phi = Qj.T@W
        W = W - Wj@Hj@Phi

lampda_sol = lambda_im + li


x=1


