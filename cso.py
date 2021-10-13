import numpy as np
from math import gamma

np.random.seed(1)

class CSO:
    def __init__(self, function, NP, D, pa, beta, Lower, Upper, N_Gen):
        '''
        PARAMETERS:
        function: A FUNCTION WHICH EVALUATES COST (OR THE FITNESS) VALUE
        NP: POPULATION SIZE
        D: TOTAL DIMENSIONS
        pa: ASSIGNED PROBABILITY
        beta: LEVY PARAMETER
        bound: AXIS BOUND FOR EACH DIMENSION
        X: PARTICLE POSITION OF SHAPE (NP,D)
        ################ EXAMPLE #####################
        If ith egg Xi = [x,y,z], n = 3, and if
        bound = [(-5,5),(-1,1),(0,5)]
        Then, x∈(-5,5); y∈(-1,1); z∈(0,5)
        ##############################################
        N_Gen: MAXIMUM ITERATION
        best: GLOBAL BEST POSITION OF SHAPE (D,1)
        '''
        self.function = function
        self.NP = NP 
        self.D = D
        self.N_Gen = N_Gen
        self.pa = pa
        self.beta = beta
        self.Lower = Lower
        self.Upper = Upper

        # X = (U-L)*rand + L (U AND L ARE UPPER AND LOWER BOUND OF X)
        # U AND L VARY BASED ON THE DIFFERENT DIMENSION OF X

        self.X = []

        for i in range(self.D):
            x = (self.Upper - self.Lower) * np.random.rand(NP,) + self.Lower 
            self.X.append(x)

        self.X = np.array(self.X).T.copy()
        print(self.X[0, :].flags)

    def update_position_1(self):
        '''
        ACTION:
        TO CALCULATE THE CHANGE OF POSITION 'X = X + rand*C' USING LEVY FLIGHT METHOD
        C = 0.01*S*(X-best) WHERE S IS THE RANDOM STEP, and β = beta (TAKEN FROM [1])
              u
        S = -----
                1/β
             |v|
        beta = 1.5
        u ~ N(0,σu) # NORMAL DISTRIBUTION WITH ZERO MEAN AND 'σu' STANDARD DEVIATION
        v ~ N(0,σv) # NORMAL DISTRIBUTION WITH ZERO MEAN AND 'σv' STANDARD DEVIATION
        σv = 1
                     Γ(1+β)*sin(πβ/2)       
        σu^β = --------------------------
                   Γ((1+β)/2)*β*(2^((β-1)/2))
        Γ IS THE GAMMA FUNCTION
        '''
        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        σu = (num/den)**(1/self.beta)
        σv = 1
        u = np.random.normal(0, σu, self.D)
        v = np.random.normal(0, σv, self.D)
        S = u/(np.abs(v)**(1/self.beta))

        # DEFINING GLOBAL BEST SOLUTION BASED ON FITNESS VALUE

        self.best = self.X[0,:].copy()

        for i in range(1, self.NP):
            self.best = self.optimum(self.best, self.X[i,:])

        Xnew = self.X.copy()
        for i in range(self.NP):
            Xnew[i,:] += np.random.randn(self.D)*0.01*S*(Xnew[i,:]-self.best) 
            self.X[i,:] = self.optimum(Xnew[i,:], self.X[i,:])

    def update_position_2(self):
        '''
        ACTION:
        TO REPLACE SOME NEST WITH NEW SOLUTIONS
        HOST BIRD CAN THROW EGG AWAY (ABANDON THE NEST) WITH FRACTION
        pa ∈ [0,1] (ALSO CALLED ASSIGNED PROBABILITY) AND BUILD A COMPLETELY 
        NEW NEST. FIRST WE CHOOSE A RANDOM NUMBER r ∈ [0,1] AND IF r < pa,
        THEN 'X' IS SELECTED AND MODIFIED ELSE IT IS KEPT AS IT IS. 
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
        PARAMETERS:
        best: GLOBAL BEST SOLUTION 'best'
        particle_x: PARTICLE POSITION
        ACTION:
        COMPARE PARTICLE'S CURRENT POSITION WITH GLOBAL BEST POSITION
        '''
        if self.function(best) > self.function(particle_x):
            best = particle_x.copy()

        return best

    def clip_X(self):
        for i in range(self.D):
            self.X[:,i] = np.clip(self.X[:,i], self.Lower, self.Upper)

    def execute(self):
        '''
        PARAMETERS:
        t: ITERATION NUMBER
        ACTION:
        AS THE NAME SUGGESTS, THIS FUNCTION EXECUTES CUCKOO SEARCH ALGORITHM
        BASED ON THE TYPE OF PROBLEM (MAXIMIZATION OR MINIMIZATION).
        NOTE: THIS FUNCTION PRINTS THE GLOBAL FITNESS VALUE FOR EACH ITERATION
        IF THE VERBOSE IS TRUE
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

