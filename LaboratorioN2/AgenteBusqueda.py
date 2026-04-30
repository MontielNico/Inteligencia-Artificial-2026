from EntornoLaberinto import LaberintoEnv

class Nodo:
    """Clase auxiliar para representar los nodos del árbol de búsqueda."""
    def __init__(self, estado, padre=None, costo=0):
        self.estado = estado    # Tupla (fila, columna)
        self.padre = padre      # Referencia al nodo anterior
        self.costo = costo      # Costo acumulado g(n)

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
    def __init__(self, entorno: LaberintoEnv):
        self.entorno = entorno
        self.nodos_expandidos = 0
        self.camino_solucion = []
        self.costo_total = 0

    def buscar(self):
        raise NotImplementedError("El algoritmo de búsqueda debe ser implementado en la clase hija.")

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
            costo_acumulado = nodo.costo + 1
            hijo = Nodo(estado=estado_sucesor, padre=nodo, costo=costo_acumulado)
            hijos.append(hijo)
            
        return hijos
