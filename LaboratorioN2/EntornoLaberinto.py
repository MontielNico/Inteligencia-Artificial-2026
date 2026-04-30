class LaberintoEnv:
    def __init__(self, maze):
        """
        Inicializa el entorno con la matriz del laberinto.
       
        """
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        # Buscamos las posiciones clave al iniciar
        self.start = self._buscar_posicion("S")
        self.goal = self._buscar_posicion("G")

    def _buscar_posicion(self, objetivo):
        """Metodo privado para encontrar coordenadas (fila, col)"""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == objetivo:
                    return (i, j)
        return None

    def es_objetivo(self, estado):
        """Verifica si el estado actual es la meta G"""
        return estado == self.goal

    def obtener_sucesores(self, estado):
        """
        Devuelve una lista de estados alcanzables desde la posicion actual.
        Solo devuelve posiciones que no sean obstaculos '#'.
        """
        fila, col = estado
        sucesores = []
        
        # Definimos los movimientos posibles: Arriba, Abajo, Izquierda, Derecha
        movimientos = [
            (1, 0), # Norte
            (-1, 0),  # Sur
            (0, -1), # Oeste
            (0, 1)   # Este
        ]

        for df, dc in movimientos:
            n_fila, n_col = fila + df, col + dc
            
            # Verificamos que este dentro de los limites y no sea obstaculo
            if 0 <= n_fila < self.rows and 0 <= n_col < self.cols:
                if self.maze[n_fila][n_col] != "#":
                    sucesores.append((n_fila, n_col))
        
        return sucesores

    def imprimir_camino(self, camino):
        """Dibuja el laberinto con el camino encontrado para el informe"""
        import copy
        mapa_resultado = copy.deepcopy(self.maze)
        for fila, col in camino:
            if mapa_resultado[fila][col] not in ["S", "G"]:
                mapa_resultado[fila][col] = "."
        
        for fila in mapa_resultado:
            print(" ".join(fila))