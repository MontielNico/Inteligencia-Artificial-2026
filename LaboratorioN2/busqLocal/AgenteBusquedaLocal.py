import random
import itertools
import copy
import math


class Nodo:
    """Clase auxiliar para representar los nodos de los tableros de busqueda"""

    def __init__(self, tablero, h):
        self.tablero = tablero
        self.h = h

    def __lt__(self, otro):
        """Permite comparar costos h de diferentes tableros"""
        return self.h < otro.h


class AgenteLocal:
    """
    Clase de busqueda local
    """

    def __init__(self, entorno):
        self.entorno = entorno

    def hill_climbing(self, max_iteraciones=1000):
        nodo_actual = Nodo(self.entorno.estado_inicial_completo(), 0)
        nodo_actual.h = self.entorno.h(nodo_actual.tablero)

        iteraciones = 0
        while iteraciones < max_iteraciones:
            if nodo_actual.h == 0:
                return nodo_actual, iteraciones

            vecinos = []  # Busqueda de vecinos
            f = random.randint(0, self.entorno.tamano - 1)
            columnas_movibles = [
                c
                for c in range(self.entorno.tamano)
                if (f, c) not in self.entorno.fijas
            ]

            if len(columnas_movibles) >= 2:
                for c1, c2 in itertools.combinations(
                    columnas_movibles, 2
                ):  # Busqueda de todos los vecinos existentes dentro del tablero de sudoku,
                    # buscando movimiento entre dos pares de numeros, respetando las cantidades
                    # de numeros diferentes dentro del tablero
                    nuevo_tablero = copy.deepcopy(nodo_actual.tablero)
                    nuevo_tablero[f][c1], nuevo_tablero[f][c2] = (
                        nuevo_tablero[f][c2],
                        nuevo_tablero[f][c1],
                    )
                    vecinos.append(Nodo(nuevo_tablero, self.entorno.h(nuevo_tablero)))

            if not vecinos:
                break

            mejor_vecino = min(vecinos)

            if mejor_vecino.h < nodo_actual.h:
                nodo_actual = mejor_vecino
            else:
                break

            iteraciones += 1

        return nodo_actual, iteraciones

    def simulated_annealing(
        self, temp_inicial=100.0, alpha=0.99, max_iteraciones=10000
    ):
        nodo_actual = Nodo(self.entorno.estado_inicial_completo(), 0)
        nodo_actual.h = self.entorno.h(nodo_actual.tablero)

        T = temp_inicial
        iteraciones = 0

        while T > 0.01 and iteraciones < max_iteraciones:
            if nodo_actual.h == 0:
                return nodo_actual, iteraciones

            # Comparación de vecinos, buscando un vecino aleatorio para hacer el calculo, si no es
            # un valor que nos sirve, entrara el valor de la probabilidad dependiendo la temperatura
            # de "empeorar" para mejorar.
            tablero_vecino = self.entorno.generar_vecino_al_azar(nodo_actual.tablero)
            nodo_vecino = Nodo(tablero_vecino, self.entorno.h(tablero_vecino))

            delta_e = nodo_actual.h - nodo_vecino.h

            if delta_e > 0:
                nodo_actual = nodo_vecino
            else:
                probabilidad = math.exp(delta_e / T)
                if random.random() < probabilidad:
                    nodo_actual = nodo_vecino

            T *= alpha
            iteraciones += 1

        return nodo_actual, iteraciones
