from AgenteBusqueda import AgenteBusqueda, Nodo
from collections import deque

"""
AGENTE DE BUSQUEDA POR PROFUNDIDAD (DFS)
"""
class AgenteDFS(AgenteBusqueda):
    def buscar(self):
        nodo_raiz = Nodo(self.entorno.start)
        if self.entorno.es_objetivo(nodo_raiz.estado):
            self.camino_solucion = nodo_raiz.obtener_camino()
            return self.camino_solucion
        frontera = deque()
        frontera.append(nodo_raiz)
        estados_en_frontera = {nodo_raiz.estado}
        explorada = set()

        while len(frontera) > 0:
            nodo = frontera.pop() #pop() para representar una pila LIFO
            estados_en_frontera.remove(nodo.estado)

            explorada.add(nodo.estado)

            for hijo in self.expandir(nodo):
                if hijo.estado not in explorada and hijo.estado not in estados_en_frontera:
                    if self.entorno.es_objetivo(hijo.estado):
                
                        self.camino_solucion = hijo.obtener_camino()
                        return self.camino_solucion
                    
                    frontera.append(hijo)
                    estados_en_frontera.add(hijo.estado)
        return None