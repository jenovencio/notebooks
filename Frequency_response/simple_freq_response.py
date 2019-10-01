import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import linalg, csc_matrix, eye

K1 = csc_matrix(np.array([[2,-1,0],[-1,1,0],[0,-1,1]])) 
M1 = eye(K1.shape[0])
f = np.array([0,0,1])

alpha = 0.01
beta = 0.001
Z1 = lambda w : K1 - w**2*M1 + 1J*w*(alpha*K1 + beta*M1)

w_array = np.linspace(0.01,2,500)
u_array = np.array(list(map(lambda w : linalg.spsolve(Z1(w),f) ,w_array)))

plt.figure()
plt.plot(w_array, np.abs(u_array),'o')
plt.title('Frequency Response - Amplitude')
plt.yscale('log')

plt.figure()
plt.plot(w_array, np.angle(u_array),'o')
plt.title('Frequency Response - Shift')

plt.show()



                         
        
 