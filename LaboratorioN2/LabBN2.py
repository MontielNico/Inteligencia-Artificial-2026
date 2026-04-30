from EntornoLaberinto import LaberintoEnv
from AgenteBFS import AgenteBFS
from AgenteDFS import AgenteDFS
from AgenteUCS import AgenteUCS

maze = [
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"],
    ["#", "S", "#", " ", " ", " ", " ", " ", "#", "G", "#"],
    ["#", " ", "#", " ", "#", "#", "#", " ", "#", " ", "#"],
    ["#", " ", "#", " ", "#", " ", " ", " ", "#", " ", "#"],
    ["#", " ", " ", " ", "#", " ", "#", "#", "#", " ", "#"],
    ["#", "#", "#", " ", "#", " ", " ", " ", " ", " ", "#"],
    ["#", " ", " ", " ", " ", " ", "#", "#", "#", "#", "#"],
    ["#", "#", "#", "#", "#", "#", "#", "#", "#", "#", "#"]
]

entorno = LaberintoEnv(maze)

agenteBfs = AgenteBFS(entorno)
agenteDfs = AgenteDFS(entorno)
agenteUcs = AgenteUCS(entorno)

camino = agenteBfs.buscar()
entorno.imprimir_camino(camino)
print("Nodos expandidos:", agenteBfs.nodos_expandidos)
print("Camino encontrado:", camino)

print("---------------------------------------------------------")

caminodfs = agenteDfs.buscar()
entorno.imprimir_camino(caminodfs)
print("Nodos expandidos:", agenteDfs.nodos_expandidos)
print("Camino encontrado:", caminodfs)

print("---------------------------------------------------------")

caminoucs = agenteUcs.buscar()
entorno.imprimir_camino(caminoucs)
print("Nodos expandidos:", agenteUcs.nodos_expandidos)
print("Camino encontrado:", caminoucs)