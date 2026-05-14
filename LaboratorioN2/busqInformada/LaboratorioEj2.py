from EntornoLaberinto import LaberintoEnv
from AgenteBusquedaInf import AgenteGreedy, AgenteAEstrella

maze = [
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
    ["#", "S", " ", " ", " ", " ", " ", " ", "#", " ", " ", " ", " ", " ", "#"],
    ["#", " ", "#", "#", "#", "#", "#", " ", "#", " ", "#", "#", "#", " ", "#"],
    ["#", " ", " ", " ", " ", " ", "#", " ", "#", " ", "#", " ", " ", " ", "#"],
    ["#", "#", "#", " ", "#", " ", "#", " ", "#", " ", "#", " ", "#", "#", "#"],
    ["#", " ", " ", " ", "#", " ", " ", " ", "#", " ", " ", " ", "#", "G", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
]

entorno = LaberintoEnv(maze)

agenteGreedy = AgenteGreedy(entorno)
agenteAEstrella = AgenteAEstrella(entorno)

solucion_greedy = agenteGreedy.buscar()
print("--- RESULTADOS GREEDY ---")
if solucion_greedy:
    camino = agenteGreedy.camino_solucion
    entorno.imprimir_camino(camino)
    print("Camino encontrado:", agenteGreedy.camino_solucion)
    print("Costo Camino Encontrado:", len(agenteGreedy.camino_solucion))
else:
    print("NO SE ENCONTRO SOLUCION. Mostrando el recorrido explorado:")
    entorno.imprimir_camino(agenteGreedy.decisiones)

print("Nodos expandidos:", agenteGreedy.nodos_expandidos)
print("Retrocesos:", agenteGreedy.cantidad_retrocesos)
print("Total de decisiones:", len(agenteGreedy.decisiones))
print("Decisiones:", agenteGreedy.decisiones)

print("\n---------------------------------------------------------\n")

solucion_a_estrella = agenteAEstrella.buscar()
print("--- RESULTADOS A* ---")
if solucion_a_estrella:
    camino = agenteAEstrella.camino_solucion
    entorno.imprimir_camino(camino)
    print("Camino encontrado:", agenteAEstrella.camino_solucion)
    print("Costo Camino Encontrado:", len(agenteAEstrella.camino_solucion))
else:
    print("NO SE ENCONTRO SOLUCION. Mostrando el recorrido explorado:")
    entorno.imprimir_camino(agenteAEstrella.decisiones)

print("Nodos expandidos:", agenteAEstrella.nodos_expandidos)
print("Retrocesos:", agenteAEstrella.cantidad_retrocesos)
print("Total de decisiones:", len(agenteAEstrella.decisiones))
print("Decisiones:", agenteAEstrella.decisiones)

