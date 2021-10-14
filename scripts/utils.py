from sklearn import cluster
from sklearn.cluster import DBSCAN
import math

def parseSeconds(seconds):
  SEC_PER_DAYS = 84600
  SEC_PER_HOURS = 3600
  SEC_PER_MINUTES = 60

  days = int(seconds // SEC_PER_DAYS)
  seconds = seconds % SEC_PER_DAYS

  hours = int(seconds // SEC_PER_HOURS)
  seconds = seconds % SEC_PER_HOURS

  minutes = int(seconds // SEC_PER_MINUTES)
  seconds = seconds % SEC_PER_MINUTES

  return f'{days}:{hours}:{minutes}:' + '{:.3f}'.format(seconds)

def pause():
  programPause = input("Press the <ENTER> key to continue...")

def get_distance(v1, v2):
  distance = 0.0
  for i in range(len(v1)):
    distance += (v2[i] - v1[i]) ** 2

  return math.sqrt(distance)

def get_epsilon_value_knn(k_value, puntos):
  mean_total_puntos = 0.0

  for (i, pto_origen) in enumerate(puntos):
    distancias = []

    for (j, ptd_destino) in enumerate(puntos):
      if i != j:
        distancias.append(get_distance(pto_origen, ptd_destino))

    sorted(distancias)

    mean_punto_origen = 0.0

    for i in range(k_value):
      mean_punto_origen += distancias[i]

    mean_punto_origen /= k_value

    mean_total_puntos += mean_punto_origen

  mean_total_puntos /= len(puntos)

  return mean_total_puntos

def clusterize_solutions(sols, min_samp):
  epsilon = get_epsilon_value_knn(min_samp, sols)
  model = DBSCAN(eps=epsilon, min_samples=min_samp)
  clusters = model.fit(X=sols)
  return clusters

