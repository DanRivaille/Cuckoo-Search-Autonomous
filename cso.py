import time
import csv
from math import gamma
import numpy as np
from scripts.utils import *

MAX_NEST = 100
MIN_NEST = 10
INCREMENTS_NEST = 2
INCREMENTS_NEST_PER_CLUSTER = 2         # Cantidad de nidos a agregar por cluster
IMPROVE_PERCENTAGE_ACCEPTED = 10        # Porcentaje de mejora aceptado para aplicar el autonomo
DIFF_CLUSTER_PERCENTAGE_ACCEPTED = 5    # Diferencia porcentual aceptado para clusters juntos

class CSO:
    def __init__(self, function, NP, D, pa, beta, Lower, Upper, N_Gen, num_function, ejecution, BKS):
        self.num_function = num_function
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
        self.F_min = 0
        self.improve_percentage = 1.0

        self.X = []
        self.fitness = []

    def sort_by_fitness(self):
        '''
        Se ordenan las soluciones y las variables relacionadas con estas, a partir
        del fitness
        '''
        # Se empaquetan los datos, para que cada murcielago tenga sus datos juntos al ordenarlos
        l = list(zip(self.X, self.fitness))

        # Se ordenan los murcielagos, a partir del valor del fitness
        ol = sorted(l, key=lambda y: y[1])

        # Se desempaquetan las listas ordenadas (llegan como tuplas)
        self.X, self.fitness = list(zip(*ol))

        # Se vuelven a pasar a listas (ya que al ordenar con zip, llegan como tuplas)
        self.fitness = list(self.fitness)
        self.X = np.array(self.X)

    def calculate_percentage(self, past_best, new_best):
        '''
        Calcula un porcentaje de mejora a partir del mejor pasado y el mejor actual
        '''
        current_difference = past_best - new_best
        percentage = (current_difference * 100) / past_best
        return percentage

    def update_improve_percentage(self, past_best):
        '''
        Actualiza el porcentaje de mejora
        '''
        self.improve_percentage = self.calculate_percentage(past_best, self.fitness[0])

    def check_improve(self, clusters_writter, past_best, iteration):
        '''
        Aplica el autonomo
        '''
        global INCREMENTS_NEST
        global INCREMENTS_NEST_PER_CLUSTER

        # Se revisa si el porcentaje de mejora es menor que el aceptado, si lo es
        # se implementan las estrategias de autoajuste
        if self.improve_percentage > IMPROVE_PERCENTAGE_ACCEPTED:
            # Si la solucion ha mejorado, y no se ha llegado al limite se decrementan los murcielagos
            if self.NP - INCREMENTS_NEST >= MIN_NEST:
                # Se decrementan la cantidad de murcielagos
                self.NP -= INCREMENTS_NEST

                # Se eliminan los peores murcielagos con sus datos de cada lista
                self.fitness = self.fitness[:-INCREMENTS_NEST]
                self.X = self.X[:-INCREMENTS_NEST]
        else:
            print(f"Improvement percetage: {round(self.improve_percentage, 2)}%  Applying self-tunning strategies")

            new_solutions = []

            # Se clusterizan las soluciones
            k = 3
            clusters, epsilon = clusterize_solutions(self.X, k)
            labels = clusters.labels_
            unique_labels = np.unique(labels)
            cant_clusters = unique_labels.shape[0]

            # Se obtiene la informacion de los clusters
            info_clusters = getInfoClusters(labels, self.fitness)

            # VER getInfoClusters PARA VER EL FORMATO DE info_clusters
            # Se guardan los logs del cluster
            for label in unique_labels:
                min_value = info_clusters[label]['min']
                max_value = info_clusters[label]['max']
                mean_cluster = info_clusters[label]['mean']
                quantity = info_clusters[label]['quantity']

                cluster_logs = f'{self.seed},{self.num_function},{self.ejecution},{iteration},{cant_clusters},{min_value},{max_value},{quantity},{mean_cluster},{epsilon},{k},{label}'
                clusters_writter.writerow(cluster_logs.split(','))

            # Sino se alcanzo el limite, se incrementa la poblacion de murcielagos
            if self.NP + (cant_clusters * INCREMENTS_NEST_PER_CLUSTER) < MAX_NEST:
                # Se obtienen las nuevas soluciones generadas (llega una lista de tuplas, que guarda
                # como primer elemento la solucion generada localmente, y como segundo elemento el indice
                # del murcielago sobre el que se genero la solucion local
                new_solutions = self.increment_cluster(clusters)

                # Se guarda la cantidad de murcielagos que se agregaron, para despues eliminar la misma cantidad
                INCREMENTS_NEST = cant_clusters * INCREMENTS_NEST_PER_CLUSTER

            # Si todos los muercielagos estan muy juntos, se reemplaza la mitad
            self.replace_cluster(clusters)

            # Si hay nuevas soluciones se agregan 
            for element in new_solutions:
                nest, index = element
                self.add_new_nest(nest, index)

            # Se actualiza el mejor fitness
            self.best_nest()

    def best_nest(self):
        '''
        Busca y actualiza el mejor nido
        '''
        j = 0
        for i in range(self.NP):
            if self.fitness[i] < self.fitness[j]:
                j = i

        self.best = self.X[j, :].copy()
        self.F_min = self.fitness[j]

    def add_new_nest(self, new_nest, index):
        '''
        Agrega un nuevo nido a la poblacion de soluciones
        '''
        # Se ingresan los datos del nuevo muercielago
        self.X = np.append(self.X, [new_nest], axis=0)
        self.fitness.append(self.function(new_nest))
        self.NP += 1

    def increment_cluster(self, clusters):
        '''
        Se incrementa la poblacion de los clusters, agregando murcielagos generados
        alrededor de los mejores de cada cluster
        '''
        x_is_modified = False
        best_nest_clusters = {l: {'index': [], 'cant': 0} for l in np.unique(clusters.labels_)}

        # Se guardan los indices de los INCREMENTS_NEST_PER_CLUSTER mejores murcielagos de cada cluster
        for index, label in enumerate(clusters.labels_):
            if best_nest_clusters[label]['cant'] < INCREMENTS_NEST_PER_CLUSTER:
                best_nest_clusters[label]['index'].append(index)
                best_nest_clusters[label]['cant'] += 1

        # Se guardan las nuevas soluciones generadas, junto con el indice del murcielago
        # sobre el cual se gener?? la solucion local
        new_solutions = []

        # Se generan INCREMENTS_NEST_PER_CLUSTER soluciones locales de los mejores murcielagos de cada cluster
        for label in best_nest_clusters:
            for index in best_nest_clusters[label]['index']:
                # Se encuentra una nueva solucion local
                new_solution = np.empty(self.D)
                new_solution = self.generate_local_solution(self.X[index])
                new_solutions.append((new_solution, index))

        return new_solutions


    def replace_cluster(self, clusters):
        '''
        Reemplaza la mitad mas mala de los clusters, con soluciones generadas aleatorias
        '''
        # Diccionario que contiene la informacion para calcular el promedio de cada cluster.
        # Para cada cluster se puede acceder a su informacion por su label,
        # como valor guarda un diccionario para organizar su informacion
        fitness_clusters = {l: {'sum':0, 'total':0} for l in np.unique(clusters.labels_)}

        # Se obtiene la suma de los fitness y el total de elementos en cada cluster
        for (index, label) in enumerate(clusters.labels_):
            fitness_clusters[label]['sum'] += self.fitness[index]
            fitness_clusters[label]['total'] += 1

        for label in fitness_clusters:
            # Se calcula el promedio
            suma = fitness_clusters[label]['sum']
            total = fitness_clusters[label]['total']
            mean_cluster = suma / total

            percentage_diff = self.calculate_percentage(self.F_min, mean_cluster)

            if -DIFF_CLUSTER_PERCENTAGE_ACCEPTED <= percentage_diff <= DIFF_CLUSTER_PERCENTAGE_ACCEPTED:
                # Se reemplaza la mitad mas mala del cluster con soluciones aleatorias usando la funcion de exploracion
                cant = total // 2

                for index in range(self.NP - 1, -1, -1):
                    if cant <= 0:
                        break

                    # Si el elemento actual pertenece al cluster que queremos repoblar
                    if clusters.labels_[index] == label:
                        self.X[index], self.fitness[index] = self.generate_random_solution(self.X[index])
                        cant -= 1

                print(percentage_diff, self.F_min - mean_cluster, self.F_min, mean_cluster, label)


    def init_cuckoo(self):
        '''
        Initialize the variables of Cuckoo Search
        '''
        np.random.seed(self.seed)

        for i in range(self.NP):
            x = (self.Upper - self.Lower) * np.random.rand(self.D,) + self.Lower
            self.X.append(x)
            self.fitness.append(self.function(x))

        self.X = np.array(self.X).copy()
        self.best = self.X[0, :].copy()
        self.F_min = self.function(self.best)

    def update_position_1(self):
        '''
        Calculate the change of position using Levy flight method
        '''
        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        ??u = (num/den)**(1/self.beta)
        ??v = 1
        u = np.random.normal(0, ??u, self.D)
        v = np.random.normal(0, ??v, self.D)
        S = u/(np.abs(v)**(1/self.beta))

        for i in range(1, self.NP):
            self.best = self.optimum(self.best, self.X[i,:])

        self.F_min = self.function(self.best)

        Xnew = self.X.copy()
        for i in range(self.NP):
            Xnew[i,:] += np.random.randn(self.D)*0.01*S*(Xnew[i,:]-self.best)
            self.X[i,:] = self.optimum(Xnew[i,:], self.X[i,:])
            self.fitness[i] = self.function(self.X[i,:])

    def update_position_2(self):
        '''
        Replace some nest with new solutions
        '''
        Xnew = self.X.copy()
        Xold = self.X.copy()
        for i in range(self.NP):
            d1,d2 = np.random.randint(0, self.NP, 2)
            for j in range(self.D):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i,j] += np.random.rand() * (Xold[d1,j]-Xold[d2,j])
            self.X[i,:] = self.optimum(Xnew[i,:], self.X[i,:])
            self.fitness[i] = self.function(self.X[i,:])

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

    def execute(self, name_logs_file='logs.csv', name_cluster_logs_file='clusters.csv', original_MH=True, interval_logs=100):
        '''
        Execute the Cuckoo Search Algorithm
        '''
        # Archivo de logs de la MH
        logs_file = open(name_logs_file, mode='w')
        logs_writter = csv.writer(logs_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        logs_writter.writerow('function,ejecution,iteration,D,NP,N_Gen,pa,beta,lower,upper,time_ms,seed,BKS,fitness,%improvement'.split(','))

        # Archivo de logs de los clusters
        clusters_file = open(name_cluster_logs_file, mode='w')
        cluster_writter = csv.writer(clusters_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        cluster_writter.writerow('seed,function,ejecution,iteration,cantClusters,min_value,max_value,cantElements,meanCluster,epsilon,k,label'.split(','))

        self.init_cuckoo()
        past_best = self.F_min

        initial_time = time.perf_counter()

        # Meteheuristic
        for t in range(self.N_Gen + 1):
            if (t % 100) == 0:
                # LUEGO DE ESTO, LAS LISTAS ESTAN ORDENADAS POR FITNESS
                self.sort_by_fitness()
                self.update_improve_percentage(past_best)

                # Logs purposes
                MH_params = f'{self.D},{self.NP},{self.N_Gen},{self.pa},{self.beta}'
                MH_params += f',{self.Lower},{self.Upper}'
                current_time = parseSeconds(time.perf_counter() - initial_time)
                log = f'{self.num_function},{self.ejecution},{t},{MH_params},{current_time},{self.seed},{self.BKS},"{self.F_min}","{self.improve_percentage}"'
                logs_writter.writerow(log.split(','))
                print('\n' + log)

                if t != 0:
                    if not original_MH:
                        # Se ajusta la cantidad de soluciones dependiendo del desempe??o
                        self.check_improve(cluster_writter, past_best, t)

                    past_best = self.F_min

            self.update_position_1()
            self.clip_X()       # Apply bounds
            self.update_position_2()
            self.clip_X()       # Apply bounds


        # Se cierran los archivos
        logs_file.close()
        clusters_file.close()

        return self.best, self.F_min


    def generate_local_solution(self, solution):
        '''
        Genera una solucion local a partir de la 'solution' ingresada
        '''
        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        ??u = (num/den)**(1/self.beta)
        ??v = 1
        u = np.random.normal(0, ??u, self.D)
        v = np.random.normal(0, ??v, self.D)
        S = u/(np.abs(v)**(1/self.beta))

        Xnew = solution.copy()
        Xnew += np.random.randn(self.D)*0.01*S*(Xnew-self.best)

        return Xnew


    def generate_random_solution(self, solution):
        '''
        Genera una solucion aleatoria
        '''
        Xnew = solution.copy()
        Xold = solution.copy()

        d1,d2 = np.random.randint(0, self.NP, 2)
        for j in range(self.D):
            r = np.random.rand()
            if r < self.pa:
                Xnew[j] += np.random.rand() * (self.X[d1,j]-self.X[d2,j])

        fitness_random_solution = self.function(Xnew)
        return Xnew, fitness_random_solution

