import time
import csv
from math import gamma
import numpy as np

np.random.seed(1)       # Just for debug

class CSO:
    def __init__(self, function, NP, D, pa, beta, Lower, Upper, N_Gen, ejecution, BKS):
        self.ejecution = ejecution
        self.BKS = BKS
        self.seed = int(time.time())

        # MH params
        self.function = function
        self.NP = NP 
        self.D = D
        self.N_Gen = N_Gen
        self.pa = pa
        self.beta = beta
        self.Lower = Lower
        self.Upper = Upper

        self.X = []

    def init_cuckoo(self):
        '''
        Initialize the variables of Cuckoo Search
        '''
        for i in range(self.D):
            x = (self.Upper - self.Lower) * np.random.rand(self.NP,) + self.Lower 
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

    def execute(self, n_fun, name_logs_file='logs.csv', interval_logs=100):
        '''
        Execute the Cuckoo Search Algorithm
        '''
        self.init_cuckoo()

        with open(name_logs_file, mode='w') as logs_file:
            initial_time = time.perf_counter()
            logs_writter = csv.writer(logs_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            logs_writter.writerow('function,ejecution,iteration,D,NP,N_Gen,pa,beta,lower,upper,time_ms,seed,BKS,fitness'.split(','))

            # Meteheuristic
            for t in range(self.N_Gen):
                if (t % 100) == 0:
                    # Logs purposes
                    MH_params = f'{self.D},{self.NP},{self.N_Gen},{self.pa}'
                    MH_params += f',{self.Lower},{self.Upper}'

                    #current_time = parseSeconds(time.perf_counter() - initial_time)
                    current_time = 0
                    log = f'{n_fun},{self.ejecution},{t},{MH_params},{current_time},{self.seed},{self.BKS},"fmin"'
                    logs_writter.writerow(log.split(','))
                    print('\n' + log)

                self.update_position_1()
                self.clip_X()       # Apply bounds
                self.update_position_2()
                self.clip_X()       # Apply bounds


        print('\nOPTIMUM SOLUTION\n  >', np.round(self.best.reshape(-1),7).tolist())
        print('\nOPTIMUM FITNESS\n  >', np.round(self.function(self.best),7))
        print()

        return self.best, self.function(self.best)


