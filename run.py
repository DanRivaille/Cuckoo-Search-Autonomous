from cso import CSO
from cec2013lsgo.cec2013 import Benchmark


def main():
    bench = Benchmark()

    num_function = 2
    ejecution = 0

    info = bench.get_info(num_function)
    BKS = info['best']
    Lower = info['lower']
    Upper = info['upper']
    D = info['dimension']
    NP = 30
    pa = 0.25
    beta = 1.5
    N_Gen = 1500
    ObjetiveFunction = bench.get_function(num_function)

    cuckoo = CSO(ObjetiveFunction, NP, D, pa, beta, Lower, Upper, N_Gen, ejecution, BKS)
    cuckoo.execute(num_function, f'Logs/function{num_function}_{ejecution}.csv')


def handle_args():
    pass


if __name__ == '__main__':
    # Argument handling
    handle_args()

    main()
