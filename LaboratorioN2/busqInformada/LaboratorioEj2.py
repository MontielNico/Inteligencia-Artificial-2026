from numpy import true_divide
from EntornoLaberinto import LaberintoEnv
from AgenteBusquedaInf import AgenteGreedy, AgenteAEstrella

maze = [
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
    ["#", "S", "#", " ", " ", " ", " ", " ", "#", "G", "#"],
    ["#", " ", "#", " ", "#", "#", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", " ", " ", "#", " ", "#"],
    ["#", " ", " ", " ", "#", " ", "#", "#", "#", " ", "#"],
    ["#", "#", "#", " ", "#", " ", " ", " ", " ", " ", "#"],
    ["#", " ", " ", " ", " ", " ", "#", "#", "#", "#", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
]

entorno = LaberintoEnv(maze)

agenteGreedy = AgenteGreedy(entorno)
agenteAEstrella = AgenteAEstrella(entorno)

if agenteGreedy.buscar() == True:
    camino = agenteGreedy.camino_solucion
    entorno.imprimir_camino(camino)
    print("Nodos expandidos:", agenteGreedy.nodos_expandidos)
    print("Camino encontrado:", agenteGreedy.camino_solucion)
else:
    print("No se encontro solucion.")
print("---------------------------------------------------------")

if agenteAEstrella.buscar() == True:
    camino = agenteAEstrella.camino_solucion
    entorno.imprimir_camino(camino)
    print("Nodos expandidos:", agenteAEstrella.nodos_expandidos)
    print("Camino encontrado:", agenteAEstrella.camino_solucion)
    print("Costo Camino Encontrado:", len(agenteAEstrella.camino_solucion))
    print("Retrocesos:", agenteAEstrella.cantidad_retrocesos)
    print("Total de decisiones:", len(agenteAEstrella.decisiones))
    print("Decisiones:", agenteAEstrella.decisiones)
else:
    print("No se encontro solucion.")
