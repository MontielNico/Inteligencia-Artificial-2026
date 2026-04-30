import heapq
from AgenteBusqueda import AgenteBusqueda, Nodo

class AgenteUCS(AgenteBusqueda):
    def buscar(self):
        nodo_raiz = Nodo(self.entorno.start)
        frontera = []


        #heapq ordena las tuplas por su primer elemento (el costo).
        #Usamos id(nodo_raiz) como segundo elemento por si hay un empate de costos
        heapq.heappush(frontera, (nodo_raiz.costo, id(nodo_raiz), nodo_raiz))

        #En lugar de un set, usamos un diccionario para los estados en frontera, así podemos guardar el estado y su costo actual: {estado: costo}

        estados_en_frontera = {nodo_raiz.estado: nodo_raiz.costo}
        
        explorada = set()

        while len(frontera) > 0:
            # Sacamos el nodo con el MENOR COSTO
            costo, _, nodo = heapq.heappop(frontera)
            
            # Si el estado extraído tiene un costo mayor al que tenemos guardado, significa que encontramos una ruta mejor antes y este es un nodo "viejo". Lo ignoramos.
            if nodo.estado not in estados_en_frontera or estados_en_frontera[nodo.estado] < costo:
                continue
                
            # Lo quitamos de la frontera porque ya lo vamos a procesar
            del estados_en_frontera[nodo.estado]
            
            if self.entorno.es_objetivo(nodo.estado):
                self.camino_solucion = nodo.obtener_camino()
                return self.camino_solucion
                
            explorada.add(nodo.estado)

            for hijo in self.expandir(nodo):
                if hijo.estado not in explorada:
                    if hijo.estado not in estados_en_frontera:
                        estados_en_frontera[hijo.estado] = hijo.costo
                        heapq.heappush(frontera, (hijo.costo, id(hijo), hijo))
                        
                    elif hijo.costo < estados_en_frontera[hijo.estado]:
                        estados_en_frontera[hijo.estado] = hijo.costo
                        heapq.heappush(frontera, (hijo.costo, id(hijo), hijo))
                        
        return None
