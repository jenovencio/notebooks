import numpy as np
import matplotlib.pyplot as plt

class jenkins():
    
    def __init__(self, k=2, alpha=0, mu=0.1):
        
        self.k = k
        self.alpha = alpha
        self.mu = mu
        self.un = 0
        
    def compute(self,delta_u,N):
        
        utrial = delta_u + self.alpha
        self.un = delta_u
        d = utrial/np.linalg.norm(utrial)
        Ft_trial = self.k*utrial
        Phi = Ft_trial - self.mu*N
        if Phi<=0:
            return Ft_trial
        
        else:
            self.alpha += Phi/self.k
            return d*self.mu*N
        
u = 2*np.sin(np.linspace(0,10,200))

plt.plot(u)

jen = jenkins()

N = 30.0
func = lambda u : jen.compute(u,N)
ft = list(map(func,u))
plt.plot(ft)