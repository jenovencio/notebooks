
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
def build_Z_dict(w,M_dict,K_dict,alpha=0.0005,beta=0.00001):
    Z_dict = {}
    for key, K in K_dict.items():
        Z_dict[key] = buildZ(w,M_dict[key],K,alpha,beta)  
    return Z_dict

width = 8.
heigh = 2.
divX=21
divY=6
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


K, f = manager.assemble_global_K_and_f()
M, _ = managerM.assemble_global_K_and_f()
B = manager.assemble_global_B()
R = manager.assemble_global_kernel()
G = manager.assemble_G()
GGT = manager.assemble_GGT()
S = manager.assemble_global_scaling()

GGT_inv = np.linalg.inv(GGT.A)
BTB_inv = np.linalg.pinv((B.T@B).A)
e = BTB_inv.diagonal() - 1/S

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

f_ = 1.0E5*L.dot(f)
#f_ = V.dot(100*np.ones(V.shape[1]))
#f_ = V[:,5]
f = Lexp.dot(f_)



class FETI_system():
    def __init__(self,Z_dict,B_dict,f_dict,tol=1.0e-3,dtype=np.float,dual_interface_algorithm='PCPG'):
        #f_dict = manager.vector2localdict(f,manager.global2local_primal_dofs)
        #self.feti_obj = SerialFETIsolver(Z_dict,B_dict,f_dict,tolerance=tol,dtype=np.complex)
        #self.mapdict = manager.local2global_primal_dofs
        self.Z_dict = Z_dict
        self.B_dict = B_dict 
        self.tol = tol
        self.feti_obj = SerialFETIsolver(self.Z_dict,self.B_dict,f_dict,
                                        tolerance=self.tol,dtype=dtype,max_int=100,
                                        dual_interface_algorithm=dual_interface_algorithm,
                                        pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-12})
        self.dtype = dtype

    def apply(self,f):

        f_dict = self.array2dict(f)
        self.feti_obj.f_dict = f_dict
        solution_obj = self.feti_obj.solve()
        u_dict = solution_obj.u_dict
        return solution_obj.displacement

    def array2dict(self,v):
        v_dict = {}
        for key, posid in mapdict.items():
            if v.ndim<2:
                v_dict[key] = v[posid]
            else:
                v_dict[key] = v[posid,:]
        return v_dict 
    
    def dict2array(self,v_dict):
        v = np.zeros(self.shape[0],dtype=self.dtype)
        for key, posid in self.mapdict.items():
            v[posid] = v_dict[key]
        return v


def array2dict(v):
    v_dict = {}
    for key, posid in mapdict.items():
        if v.ndim<2:
            v_dict[key] = v[posid]
        else:
            v_dict[key] = v[posid,:]
    return v_dict 


mapdict = manager.local2global_primal_dofs
usize = manager.primal_size
def create_FETI_operator(Z_dict,B_dict,dtype=np.complex,dual_interface_algorithm='PCPG'):
    FETIobj = FETI_system(Z_dict,B_dict,f_dict,dtype=dtype,dual_interface_algorithm=dual_interface_algorithm)
    return LO(lambda x : FETIobj.apply(x), shape=(usize,usize),dtype=dtype)

w = 0.0
Z_dict = build_Z_dict(w,M_dict,K_dict)
FETIop = create_FETI_operator(Z_dict,B_dict)
#fc = f + 0.0*1J*f
#fc1 = 2*f + 0.0*1J*f
#e1 = FETIop.dot(fc)
#e2 = FETIop.dot(fc1)
#e3 = FETIop.dot(np.array([fc,fc1]).T)
#error1 = np.abs(e1 - e3[:,0])
#error2 = np.abs(e2 - e3[:,1])



@timing
def primal_sys(w_list):
    u_list = []
    for w in w_list:
        #Z_dict = build_Z_dict(w,M_dict,K_dict)
        #u0, u_dict = system(Z_dict,tol=1.0e-12)
        Z = buildZ(w,Mp,Kp)
        usol = sparse.linalg.spsolve(Z,f_)
        u0 = Lexp.dot(usol)
        u_list.append(u0)
        #u_obj_list.append(np.abs(B_obj.dot(u_dict[3])[1]))
    return u_list

@timing
def proj_sys(w_list):
    fp = P.T.dot(f)
    u_list = []
    u_init = np.zeros(K.shape[0])
    update = True
    number_of_iterations_list = []
    count=0
    for i in range(2*len(w_list)):
        w = w_list[count]
        Z = buildZ(w,M,K)
        Zp = P.T.dot(Z.dot(P))
        Z_action = LO(lambda x : Zp.dot(x), shape=Z.shape, dtype=Z.dtype)

        if update:
            #lu = sparse.linalg.splu(Z)
            #Dp = LO(lambda x : P.T.dot(lu.solve(P.dot(x))), shape=Z.shape, dtype=Z.dtype)
            #u_init = Dp.dot(f)
            pass
            
        error = Z_action.dot(u_init) - fp
        #usol, info = sparse.linalg.gmres(Z_action,fp,x0=u_init, tol=1.0E-6)
        usol, info = sparse.linalg.lgmres(Z_action,fp,x0=u_init, tol=1.0E-6)
        number_of_iterations_list.append(Z_action.count)
        error2 = Z_action.dot(usol) - fp
        
        if info!=0:
            update=True
            if np.linalg.norm(error)>np.linalg.norm(error2):
                u_init = usol
            else:
                u_list.append(u_init)
                count+=1
        else:
            update=False
            u_init = usol
            u_list.append(usol)
            count+=1
            
        if w==w_list[-1]:
            break

    return u_list, number_of_iterations_list 


@timing
def proj_feti_sys(w_list):
    u_list = []
    u_init = np.zeros(K.shape[0])
    update = True
    count=0
    fc = f + 0.0*1J*f
    fp = P.T.dot(fc)
    number_of_iterations_list = []
    for i in range(2*len(w_list)):
        w = w_list[count]
        Z = buildZ(w,M,K)
        Zp = P.T.dot(Z.dot(P))
        if update:
            Z_dict = build_Z_dict(w,M_dict,K_dict,alpha=0.0,beta=0.0)
            FETIop = create_FETI_operator(Z_dict,B_dict,dtype=np.complex)
            F = sparse.linalg.LinearOperator(shape=FETIop.shape,dtype=FETIop.dtype, matvec = lambda x : P.dot(FETIop.dot(x)))
            u_init = F.dot(fp)
            
        Z_action = LO(lambda x : Zp.dot(x), shape=Z.shape, dtype=Z.dtype)
        error = Z_action.dot(u_init) - fp
        #usol, info = sparse.linalg.lgmres(Z_action,fp,x0=u_init,maxiter=60,tol=1.0E-6)
        usol, info = sparse.linalg.lgmres(Z_action,fp,x0=u_init,maxiter=300,tol=1.0E-6,M=F)
        number_of_iterations_list.append(Z_action.count)
        #usol = P.dot(FETIop.dot(fp))
        error = np.linalg.norm(Z_action.dot(usol) - fp)/np.linalg.norm(fp)

        #info=0
        #Z_action.count
        if info!=0:
            update=True
            continue
        
        elif Z_action.count>30:
            update=True

        else:
            update=False

        u_init = usol
        count+=1

        u0 = usol
        
        u_list.append(u0)
        
        if w==w_list[-1]:
            break

    return u_list, number_of_iterations_list




@timing
def proj_feti_gmres_sys(w_list):
    u_list = []
    u_init = np.zeros(K.shape[0])
    update = True
    count=0
    fc = f + 0.0*1J*f
    fp = P.T.dot(fc)
    number_of_iterations_list = []
    num_of_updates = 0
    for i in range(2*len(w_list)):
        w = w_list[count]
        Z = buildZ(w,M,K)
        Zp = P.T.dot(Z.dot(P))
        if update:
            Z_dict = build_Z_dict(w,M_dict,K_dict)
            FETIop = create_FETI_operator(Z_dict,B_dict,dtype=np.complex,dual_interface_algorithm='PGMRES')
            F = sparse.linalg.LinearOperator(shape=FETIop.shape,dtype=FETIop.dtype, matvec = lambda x : P.dot(FETIop.dot(x)))
            u_init = F.dot(fp)
            num_of_updates +=1
            
        Z_action = LO(lambda x : Zp.dot(x), shape=Z.shape, dtype=Z.dtype)
        error = Z_action.dot(u_init) - fp
        #usol, info = sparse.linalg.lgmres(Z_action,fp,x0=u_init,maxiter=60,tol=1.0E-6)
        usol, info = sparse.linalg.gmres(Z_action,fp,x0=u_init,maxiter=300,tol=1.0E-6,M=F)
        number_of_iterations_list.append(Z_action.count)
        #usol = P.dot(FETIop.dot(fp))
        error = np.linalg.norm(Z_action.dot(usol) - fp)/np.linalg.norm(fp)

        #info=0
        #Z_action.count
        if info!=0:
            update=True
            continue

        elif Z_action.count>15:
            update=True

        else:
            update=False
        u_init = usol
        count+=1

        u0 = usol
        
        u_list.append(u0)
        
        if w==w_list[-1]:
            break

    return u_list, number_of_iterations_list



@timing
def feti_sys(w_list):
    u_list = []
    u_init = np.zeros(K.shape[0])
    update = True
    count=0
    fc = f + 0.0*1J*f
    fp = P.T.dot(fc)
    for i in range(2*len(w_list)):
        w = w_list[count]
        #Z = buildZ(w,M,K,alpha=0.0001)
        Zp = buildZ(w,Mp,Kp)
        usol = sparse.linalg.spsolve(Zp,L.dot(fp))
        u0 = Lexp.dot(usol)
        
        Z_dict = build_Z_dict(w,M_dict,K_dict)
        FETIop = create_FETI_operator(Z_dict,B_dict,dtype=np.complex)
        F = sparse.linalg.LinearOperator(shape=FETIop.shape,dtype=FETIop.dtype, matvec = lambda x : P.dot(FETIop.dot(x)))
            
        u_dual = F.dot(fp)
        u_list.append(u_dual)
        error = np.abs(u_dual - u0)
        count+=1
        
        if w==w_list[-1]:
            break

    return u_list


@timing
def feti_gmres_sys(w_list):
    u_list = []
    u_init = np.zeros(K.shape[0])
    update = True
    count=0
    fc = f + 0.0*1J*f
    fp = P.T.dot(fc)
    for i in range(2*len(w_list)):
        w = w_list[count]
        #Z = buildZ(w,M,K,alpha=0.0001)
        Zp = buildZ(w,Mp,Kp)
        Z = buildZ(w,M,K)
        Zinv = sparse.linalg.inv(Z)
        d = B@Zinv@fp
        F  = B@Zinv@B.T 
        lsol = sparse.linalg.spsolve(F,d) 
        u = Zinv@(fp - B.T.dot(lsol))
        usol = sparse.linalg.spsolve(Zp,L.dot(fp))
        u0 = Lexp.dot(usol)
        
        Z_dict = build_Z_dict(w,M_dict,K_dict)
        '''
        f_dict = array2dict(fp)
        feti_obj = SerialFETIsolver(Z_dict,B_dict,f_dict,
                                        tolerance=1.0E-12,dtype=np.complex,max_int=100,
                                        dual_interface_algorithm='PGMRES',pseudoinverse_kargs={'method':'splusps','tolerance':1.0E-12})

        feti_obj.manager._build_local_matrix()
        F2 = feti_obj.manager.assemble_global_F()
        d2 = feti_obj.manager.assemble_global_d()
        '''

        FETIop = create_FETI_operator(Z_dict,B_dict,dtype=np.complex,dual_interface_algorithm='PGMRES')
        F = sparse.linalg.LinearOperator(shape=FETIop.shape,dtype=FETIop.dtype, matvec = lambda x : P.dot(FETIop.dot(x)))
            
        u_dual = F.dot(fp)
        u_list.append(u_dual)
        error = np.abs(u_dual - u0)/np.linalg.norm(u0)
        count+=1
        
        if w==w_list[-1]:
            break

    return u_list

def plot(w_list,u_list,dof_id=121,title=''):
    plt.figure()
    plt.plot(w_list,np.abs(np.array(u_list)[:,dof_id]),'o-')
    plt.yscale('log')
    plt.title(title)

@timing
def iter_primal_sys(w_list):
    u_list = []
    update = True
    u_init = np.zeros(Kp.shape[0])
    max_int = len(w_list)*2
    count = 0
    number_of_iterations_list = []
    num_of_updates = 0
    for i in range(max_int):
        w = w_list[count]
        Z = buildZ(w,Mp,Kp)
        Z_action = LO(lambda x : Z.dot(x), shape=Z.shape, dtype=Z.dtype)
        if update:
            Zprec = Z
            lu = sparse.linalg.splu(Zprec)
            Zprec_inv = sparse.linalg.LinearOperator(shape=Zprec.shape, matvec = lambda x : lu.solve(x))
            u_init = Zprec_inv.dot(f_)
            num_of_updates += 1

        usol, info = sparse.linalg.gmres(Z_action,f_,x0=u_init,M=Zprec_inv,maxiter=300)
        #usol, info = sparse.linalg.cg(Z_action,f_)

        number_of_iterations_list.append(Z_action.count)
        if info>0:
            update = True
            continue

        elif Z_action.count>10: 
            update = True
            u_init = usol
            count+=1
        else:
            update = False
            u_init = usol
            count+=1

        u0 = Lexp.dot(usol)
        u_list.append(u0)
        #u_obj_list.append(np.abs(B_obj.dot(u_dict[3])[1]))
        if w == w_list[-1]:
            break
    return u_list, number_of_iterations_list


@timing
def iter_primal_sys_without_precond(w_list):
    u_list = []
    update = True
    u_init = np.zeros(Kp.shape[0])
    max_int = len(w_list)*2
    count = 0
    number_of_iterations_list = []
    for i in range(max_int):
        w = w_list[count]
        Z = buildZ(w,Mp,Kp)
        Z_action = LO(lambda x : Z.dot(x), shape=Z.shape, dtype=Z.dtype)
        
        usol, info = sparse.linalg.lgmres(Z_action,f_,x0=u_init,maxiter=300)
        #usol, info = sparse.linalg.cg(Z_action,f_)

        number_of_iterations_list.append(Z_action.count)
        u_init =  usol
        count+=1

        u0 = Lexp.dot(usol)
        u_list.append(u0)
        #u_obj_list.append(np.abs(B_obj.dot(u_dict[3])[1]))
        if w == w_list[-1]:
            break
    return u_list, number_of_iterations_list

@timing
def iter_proj_sys(w_list):
    u_list = []
    update = True
    u_init = np.zeros(Kp.shape[0])
    max_int = len(w_list)*2
    count = 0
    number_of_iterations_list = []
    for i in range(max_int):
        w = w_list[count]
        Z = buildZ(w,Mp,Kp)
        Z_action = LO(lambda x : Z.dot(x), shape=Z.shape, dtype=Z.dtype)
        if update:
            Zprec = Z
            lu = sparse.linalg.splu(Zprec)
            Zprec_inv = sparse.linalg.LinearOperator(shape=Zprec.shape, matvec = lambda x : lu.solve(x))
            u_init = Zprec_inv.dot(f_)

        usol, info = sparse.linalg.cg(Z_action,f_,x0=u_init,M=Zprec_inv,maxiter=30)
        #usol, info = sparse.linalg.cg(Z,f_)
        number_of_iterations_list.append(Z_action.count)
        if info>0:
            update = True
            continue

        else:
            update = False
            u_init = usol
            count+=1

        u0 = Lexp.dot(usol)
        u_list.append(u0)
        #u_obj_list.append(np.abs(B_obj.dot(u_dict[3])[1]))
        if w == w_list[-1]:
            break
    return u_list, number_of_iterations_list

if False:
    mapdict = manager.local2global_primal_dofs
    usize = manager.primal_size
    A = ParallelMatrix(K_dict,mapdict,shape=(usize,usize),dtype=manager.dtype)

    @timing
    def t1():
        e1 =A.dot(f)
        return e1

    @timing
    def t2():
        e2 = K.dot(f)
        return e2

w_list = np.linspace(1.0,1000,200)

    
#u_list = primal_sys(w_list)
#plot(w_list,u_list,title='Primal')

"""
u_list, number_of_iterations_list = iter_primal_sys_without_precond(w_list)
plot(w_list,u_list,title='Iterative Primal without precond')

plt.figure()
plt.plot(number_of_iterations_list,'o')
plt.title('iteration Primal without precond')


u_list, number_of_iterations_list = iter_primal_sys(w_list)
plot(w_list,u_list,title='Iterative Primal')

plt.figure()
plt.plot(number_of_iterations_list,'o')
plt.title('iteration Primal with preconditioner')
"""

#u_list, number_of_iterations_list = proj_sys(w_list)
#plot(w_list,u_list,title='Projected')

#plt.figure()
#plt.plot(number_of_iterations_list,'o')
#plt.title('iteration Proj without precond')


#u_list, number_of_iterations_list = proj_feti_sys(w_list)
#plot(w_list,u_list,title='FETI-Projected')

#plt.figure()
#plt.plot(number_of_iterations_list,'o')
#plt.title('iteration Proj FETI precond')


u_list, number_of_iterations_list = proj_feti_gmres_sys(w_list)
plot(w_list,u_list,title='Projected-FETI-GMRES')

plt.figure()
plt.plot(number_of_iterations_list,'o')
plt.title('iteration FETI-GMRES')

#u_list, number_of_iterations_list = iter_proj_sys(w_list)
#plot(w_list,u_list,title='Projected')

#plt.figure()
#plt.plot(number_of_iterations_list,'o')
#plt.title('iteration Projected-GMRES')

#u_list = feti_sys(w_list)
#plot(w_list,u_list,title='FETI')

#u_list = feti_gmres_sys(w_list)
#plot(w_list,u_list,title='FETI-GMRES')



plt.show()
x=1

