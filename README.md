# Cuckoo-Search-Autonomous
Implementacion del algoritmo Cuckoo Search estandar con modificaciónes propuesto por XS Yang, S Deb - 2009

Se agrega la habilidad para auto-ajustar sus parametros dependiendo del desempeño que tenga en la ejecucion logrando así una poblacion dinámica.
Se agregaron las funciones CEC 2021 (2013) para el benchmarking.

Se uso como base el [repositorio](https://github.55860.com/ujjwalkhandelwal/cso_cuckoo_search_optimization) escrito por ujjwalkhandelwal, se realizaron algunas traducciones y otros cambios menores.

## Install
Para correr el algoritmo se tienen que instalar algunas dependencias entre otras cosas, los pasos serian los siguientes:

1. Crear el ambiente virtual:
```
$ python3 -m venv ./venv
```

2. Instalar las dependecias:
```
$ pip3 install -r requeriments.txt
```

3. Crear la carpeta de logs y logs de los clusters:
```
$ mkdir Logs
$ mkdir Logs/clusters
```
5. Probar que funcione:
```
$ source venv/bin/activate
(venv) $ python3 run.py
