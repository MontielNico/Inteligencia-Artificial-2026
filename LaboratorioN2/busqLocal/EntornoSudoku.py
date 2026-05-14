import random
import math
import copy


class EntornoSudoku:
    def __init__(self, tablero_inicial):
        """
        Inicializa el entorno con el tablero inicial
        """

        self.tablero_inicial = tablero_inicial
        self.tamano = len(tablero_inicial)
        self.tamano_bloque = int(math.sqrt(self.tamano))  # 2 para 4x4, 3 para 9x9

        self.fijas = set()
        for f in range(self.tamano):
            for c in range(self.tamano):
                if tablero_inicial[f][c] != 0:
                    self.fijas.add((f, c))

    def estado_inicial_completo(self):
        """
        Completa el tablero inicial con los valores restantes
        """
        tablero = copy.deepcopy(self.tablero_inicial)
        for f in range(self.tamano):
            numeros_presentes = set(tablero[f])
            numeros_faltantes = list(set(range(1, self.tamano + 1)) - numeros_presentes)
            random.shuffle(numeros_faltantes)
            for c in range(self.tamano):
                if tablero[f][c] == 0:
                    tablero[f][c] = numeros_faltantes.pop()
        return tablero

    def h(self, tablero):
        """
        Calculo del respectivo valor h, el puntaje de vecinos adyacentes
        """
        conflictos = 0
        for c in range(self.tamano):
            columna = [tablero[f][c] for f in range(self.tamano)]
            conflictos += self.tamano - len(set(columna))

        for inicio_f in range(0, self.tamano, self.tamano_bloque):
            for inicio_c in range(0, self.tamano, self.tamano_bloque):
                bloque = []
                for f in range(self.tamano_bloque):
                    for c in range(self.tamano_bloque):
                        bloque.append(tablero[inicio_f + f][inicio_c + c])
                conflictos += self.tamano - len(set(bloque))
        return conflictos

    def generar_vecino_al_azar(self, tablero):
        """
        Elección aleatorea de un vecino adyacente al mismo, para su respectivo calculo
        """
        nuevo_tablero = copy.deepcopy(tablero)
        f = random.randint(0, self.tamano - 1)
        columnas_movibles = [c for c in range(self.tamano) if (f, c) not in self.fijas]

        if len(columnas_movibles) >= 2:
            c1, c2 = random.sample(columnas_movibles, 2)
            nuevo_tablero[f][c1], nuevo_tablero[f][c2] = (
                nuevo_tablero[f][c2],
                nuevo_tablero[f][c1],
            )

        return nuevo_tablero

    def imprimir_tablero(self, tablero):
        """Imprime el tablero con separadores de cuadrantes dinámicos"""
        # Calcula el ancho de la línea horizontal superior/inferior
        linea_horizontal = "-" * (self.tamano * 3 + self.tamano_bloque - 1)
        print(linea_horizontal)

        for f in range(self.tamano):
            fila_str = ""
            for c in range(self.tamano):
                val = tablero[f][c]
                fila_str += f"{val if val != 0 else '.'}  "
                # Agrega la barra vertical | al terminar un bloque (menos en el borde final)
                if (c + 1) % self.tamano_bloque == 0 and c != self.tamano - 1:
                    fila_str += "|  "

            print(fila_str)
            # Agrega la línea horizontal al terminar un bloque de filas
            if (f + 1) % self.tamano_bloque == 0:
                print(linea_horizontal)
