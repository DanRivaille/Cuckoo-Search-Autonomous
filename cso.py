import numpy as np
from math import gamma

np.random.seed(1)       # Just for debug

class CSO:
    def __init__(self, function, NP, D, pa, beta, Lower, Upper, N_Gen):
        '''
        PARAMETERS:
        function: Objective function
        NP: Population size
        D: Total dimensions
        pa: Assigned probability
        beta: Levy parameter
        Lower: Lower bound
        Upper: Upper bound
        N_Gen: Maximum iterations
        '''
        self.function = function
        self.NP = NP 
        self.D = D
        self.N_Gen = N_Gen
        self.pa = pa
        self.beta = beta
        self.Lower = Lower
        self.Upper = Upper

        self.X = []

        for i in range(self.D):
            x = (self.Upper - self.Lower) * np.random.rand(NP,) + self.Lower 
            self.X.append(x)

        self.X = np.array(self.X).T.copy()

    def update_position_1(self):
        '''
        Calculate the change of position using Levy flight method
        '''
        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        σu = (num/den)**(1/self.beta)
        σv = 1
        u = np.random.normal(0, σu, self.D)
        v = np.random.normal(0, σv, self.D)
        S = u/(np.abs(v)**(1/self.beta))

        self.best = self.X[0,:].copy()

        for i in range(1, self.NP):
            self.best = self.optimum(self.best, self.X[i,:])

        Xnew = self.X.copy()
        for i in range(self.NP):
            Xnew[i,:] += np.random.randn(self.D)*0.01*S*(Xnew[i,:]-self.best) 
            self.X[i,:] = self.optimum(Xnew[i,:], self.X[i,:])

    def update_position_2(self):
        '''
        Replace some nest with new solutions
        '''
        Xnew = self.X.copy()
        Xold = self.X.copy()   
        for i in range(self.NP):
            d1,d2 = np.random.randint(0,5,2)
            for j in range(self.D):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i,j] += np.random.rand()*(Xold[d1,j]-Xold[d2,j]) 
            self.X[i,:] = self.optimum(Xnew[i,:], self.X[i,:])
    
    def optimum(self, best, particle_x):
        '''
        Compare particle's current position with global best position
        '''
        if self.function(best) > self.function(particle_x):
            best = particle_x.copy()

        return best

    def clip_X(self):
        '''
        Apply the bounds
        '''
        for i in range(self.D):
            self.X[:,i] = np.clip(self.X[:,i], self.Lower, self.Upper)

    def execute(self):
        '''
        Execute the Cuckoo Search Algorithm
        '''
        for t in range(self.N_Gen):
            self.update_position_1()
            self.clip_X()       # Apply bounds
            self.update_position_2()
            self.clip_X()       # Apply bounds

            if (t % 100) == 0:
                print(t)
                print(self.best[:10])
                print(self.function(self.best))
                print()

        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1),7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.function(self.best),7))
        print()

        return self.best, self.function(self.best)


