from pyfeti.src.utils import DomainCreator, dict2dfmap, create_selection_operator, DofManager, save_object
from pyfeti.src.feti_solver import SerialFETIsolver, SolverManager
from pyfeti.cases.case_generator import FETIcase_builder
from scipy import sparse
import amfe
import numpy as np
import matplotlib.pyplot as plt
import copy


mesh_file = 'meshes/3d_simple_bladed_disk_24_sector.msh'
mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_course_v3.inp'
mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_1524_nodes.inp'
mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_2649_nodes.inp'
mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_512_nodes.inp'
#mesh_file = 'meshes/3D_simple_bladed_disk_24_sectors_16480_nodes.inp'
scale = 1000
m1 = amfe.Mesh()
#m1.import_msh(mesh_file)
m1.import_inp(mesh_file,scale)
save_object(m1,'3D_simple_bladed_disk_24_sectors_512_nodes.pkl')

ax1 = amfe.plot3Dmesh(m1)
bc = np.array([-200,200])
ax1.set_xlim(bc)
ax1.set_ylim(bc)
ax1.set_zlim(bc)
plt.show()



x=1
