import sys
from cec2013lsgo.cec2013 import Benchmark
from cso import CSO

ORIGINAL_MH = True

INITIAL_FUNCTION = 1
LAST_FUNCTION = 15

INITIAL_EJECUTION = 1
LAST_EJECUTION = 31

def main():
    bench = Benchmark()

    for num_function in range(INITIAL_FUNCTION, LAST_FUNCTION + 1):
        info = bench.get_info(num_function)
        print(f'\nFunction {num_function}: {info}')

        for ejecution in range(INITIAL_EJECUTION, LAST_EJECUTION + 1):
            BKS = info['best']
            Lower = info['lower']
            Upper = info['upper']
            D = info['dimension']
            NP = 30
            pa = 0.25
            beta = 1.5
            N_Gen = 2500
            ObjetiveFunction = bench.get_function(num_function)

            cuckoo = CSO(ObjetiveFunction, NP, D, pa, beta, Lower, Upper, N_Gen, ejecution, BKS)
            cuckoo.execute(num_function, f'Logs/function{num_function}_{ejecution}.csv', ORIGINAL_MH)


def handle_args():
    """
    Funcion que maneja los argumentos provenientes de la linea de comandos, las
    opciones validas son:
    -f, --function <number>              Ejecuta solo la funcion numero 'number'
    -F, --functions-range <init>:<last>  Ejecuta de la funcion 'init' hasta 'last'
    -e, --ejecution <number>             Ejecuta solo 'number' ejecuciones
    -E, --ejecutions-range <init>:<last> Ejecuta desde 'init' hasta 'last' ejecuciones
    -A, --autonomous                     Ejecuta la metaheuristica con el autonomo
    -O, --original                       Ejecuta la metaheuristica original
    -h                                   Muestra los comandos disponibles
    """
    global INITIAL_FUNCTION
    global LAST_FUNCTION
    global INITIAL_EJECUTION
    global LAST_EJECUTION
    global ORIGINAL_MH

    cant_args = len(sys.argv)

    if cant_args != 0:
        current_arg_index = 1

    while current_arg_index < cant_args:
        current_arg = sys.argv[current_arg_index]

        if '-f' == current_arg or '--function' == current_arg:
            current_arg_index += 1
            INITIAL_FUNCTION = int(sys.argv[current_arg_index])
            LAST_FUNCTION = INITIAL_FUNCTION

        elif '-F' == current_arg or '--functions-range' == current_arg:
            current_arg_index += 1
            range_functions = sys.argv[current_arg_index].split(':')
            INITIAL_FUNCTION = int(range_functions[0])
            LAST_FUNCTION = int(range_functions[1])

        elif '-e' == current_arg or '--ejecution' == current_arg:
            current_arg_index += 1
            INITIAL_EJECUTION = int(sys.argv[current_arg_index])
            LAST_EJECUTION = INITIAL_EJECUTION

        elif '-E' == current_arg or '--ejecutions-range' == current_arg:
            current_arg_index += 1
            range_ejecutions = sys.argv[current_arg_index].split(':')
            INITIAL_EJECUTION = int(range_ejecutions[0])
            LAST_EJECUTION = int(range_ejecutions[1])

        elif '-h' == current_arg:
            help_text = "-f, --function <number>              Ejecuta solo la funcion numero 'number'"
            help_text += "\n-F, --functions-range <init>:<last>  Ejecuta de la funcion 'init' hasta 'last'"
            help_text += "\n-e, --ejecution <number>             Ejecuta solo 'number' ejecuciones"
            help_text += "\n-E, --ejecutions-range <init>:<last> Ejecuta desde 'init' hasta 'last' ejecuciones"
            help_text += "\n-A, --autonomous                     Ejecuta la metaheuristica con el autonomo"
            help_text += "\n-O, --original                       Ejecuta la metaheuristica original (Opcion por defecto)"
            help_text += "\n-h                                   Muestra los comandos disponibles"
            print(help_text)

        elif '-A' == current_arg or '--autonomous' == current_arg:
            ORIGINAL_MH = False

        elif '-O' == current_arg or '--original' == current_arg:
            ORIGINAL_MH = True

        current_arg_index += 1


if __name__ == '__main__':
    # Argument handling
    handle_args()

    main()
