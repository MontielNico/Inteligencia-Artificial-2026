from EntornoLaberinto import LaberintoEnv
import heapq


class Nodo:
    """Clase auxiliar para representar los nodos del árbol de búsqueda."""

    def __init__(self, estado, padre=None, costo_g=0, costo_h=0, heuristica="Greedy"):
        self.estado = estado  # Tupla (fila, columna)
        self.padre = padre  # Referencia al nodo anterior
        self.costo_g = costo_g  # Costo acumulado desde el nodo inicial
        self.costo_h = costo_h  # Heurística desde el nodo actual hasta el objetivo

        if heuristica == "Greedy":
            self.costo_f = costo_h  # En Greedy solo se considera la heurística
        else:  # En A* se considera el costo total f(n) = g(n) + h(n)
            self.costo_f = costo_g + costo_h

    def __lt__(self, otro):
        """Permite comparar nodos por su costo para estructuras de datos como colas de prioridad."""
        return self.costo_f < otro.costo_f

    def obtener_camino(self):
        """Reconstruye el camino desde el nodo inicial hasta este nodo."""
        camino = []
        actual = self
        while actual is not None:
            camino.append(actual.estado)
            actual = actual.padre
        return camino[::-1]  # Invertir el camino para que vaya de inicio a fin


class AgenteBusqueda:
    """
    Clase padre para los agentes de búsqueda.
    Define la estructura básica, contadores y métodos comunes.
    """

    def __init__(self, entorno: LaberintoEnv, heuristica="Greedy"):
        self.entorno = entorno
        self.nodos_expandidos = 0
        self.cantidad_retrocesos = 0
        self.camino_solucion = []
        self.decisiones = []
        self.heuristica = heuristica

    def buscar(self):
        raise NotImplementedError(
            "El algoritmo de búsqueda debe ser implementado en la clase hija."
        )

    def calculo_manhattan(self, nodo: Nodo):
        """
        Calcula la distancia Manhattan entre el estado actual y el objetivo.
        """
        x1, y1 = nodo.estado
        x2, y2 = self.entorno.goal
        return abs(x1 - x2) + abs(y1 - y2)

    def expandir(self, nodo: Nodo):
        """
        Expande el nodo actual generando los nodos hijos válidos.
        Lleva el control de cuántos nodos se han expandido en total.
        """
        self.nodos_expandidos += 1
        hijos = []

        # Obtenemos los estados sucesores (coordenadas adyacentes válidas)
        estados_sucesores = self.entorno.obtener_sucesores(nodo.estado)

        for estado_sucesor in estados_sucesores:
            # Asumimos que moverse de una celda a otra adyacente tiene un costo de paso = 1
            costo_acumulado = nodo.costo_g + 1
            hijo = Nodo(
                estado=estado_sucesor,
                padre=nodo,
                costo_g=costo_acumulado,
                costo_h=self.calculo_manhattan(Nodo(estado=estado_sucesor)),
                heuristica=self.heuristica,
            )
            hijos.append(hijo)

        return hijos


class AgenteGreedy(AgenteBusqueda):
    def __init__(self, entorno: LaberintoEnv):
        super().__init__(entorno, heuristica="Greedy")

    def buscar(self):
        # 1. Preparación de la raíz
        nodo_raiz = Nodo(
            estado=self.entorno.start,
            costo_h=self.calculo_manhattan(Nodo(estado=self.entorno.start)),
            heuristica=self.heuristica,
        )

        # 2. Cola de prioridad y Memoria simple (Set)
        frontera = []
        heapq.heappush(frontera, nodo_raiz)

        visitados = set()
        visitados.add(nodo_raiz.estado)

        ultimo_nodo_revisado = None

        # 3. Bucle de Búsqueda
        while frontera:
            nodo_actual = heapq.heappop(frontera)
            self.decisiones.append(nodo_actual.estado)

            # Registro de retrocesos (saltos)
            if (
                ultimo_nodo_revisado is not None
                and nodo_actual.padre != ultimo_nodo_revisado
            ):
                self.cantidad_retrocesos += 1
            ultimo_nodo_revisado = nodo_actual

            # Test de Objetivo
            if self.entorno.es_objetivo(nodo_actual.estado):
                self.camino_solucion = nodo_actual.obtener_camino()
                return True

            # Expandir y filtrar
            for hijo in self.expandir(nodo_actual):
                if hijo.estado not in visitados:
                    visitados.add(hijo.estado)
                    heapq.heappush(frontera, hijo)

        return False


class AgenteAEstrella(AgenteBusqueda):
    def __init__(self, entorno: LaberintoEnv):
        super().__init__(entorno, heuristica="A*")

    def buscar(self):

        frontera = []
        nodo_raiz = Nodo(
            self.entorno.start,
            costo_h=self.calculo_manhattan(Nodo(estado=self.entorno.start)),
            heuristica=self.heuristica,
        )
        heapq.heappush(frontera, nodo_raiz)
        ultimo_explorado = None
        historial_nodos = {nodo_raiz.estado: nodo_raiz.costo_g}

        while frontera:
            nodo_actual = heapq.heappop(frontera)
            self.decisiones.append(nodo_actual.estado)

            if ultimo_explorado is not None and nodo_actual.padre != ultimo_explorado:
                self.cantidad_retrocesos += 1
            ultimo_explorado = nodo_actual

            # Test de Objetivo
            if self.entorno.es_objetivo(nodo_actual.estado):
                self.camino_solucion = nodo_actual.obtener_camino()
                return True

            # Expandimos el nodo actual y filtramos a los hijos
            for hijo in self.expandir(nodo_actual):
                # REGLA DE ORO: Solo procesamos el hijo si nunca pisamos esa baldosa
                # O si este nuevo camino hacia esa baldosa es más barato que el récord anterior
                if (
                    hijo.estado not in historial_nodos
                    or hijo.costo_g < historial_nodos[hijo.estado]
                ):
                    historial_nodos[hijo.estado] = (
                        hijo.costo_g
                    )  # Actualizamos el récord
                    heapq.heappush(frontera, hijo)  # Lo mandamos a la cola de prioridad

        return False  # Fracaso: Nos quedamos sin opciones y no llegamos a la meta
