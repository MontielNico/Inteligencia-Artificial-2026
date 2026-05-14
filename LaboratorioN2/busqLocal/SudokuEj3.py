from EntornoSudoku import EntornoSudoku
from AgenteBusquedaLocal import AgenteLocal

sudoku_4x4 = [[0, 1, 0, 4], [0, 0, 3, 0], [0, 2, 0, 0], [0, 0, 0, 0]]


sudoku_9x9 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# =========================================
# SUDOKU 4x4

entorno = EntornoSudoku(sudoku_4x4)
agente = AgenteLocal(entorno)

print("ESTADO INICIAL 4x4")
entorno.imprimir_tablero(sudoku_4x4)

# 1. Demostración Visual
print("\nRESULTADO HILL CLIMBING (1 Corrida)")
nodo_hc, iter_hc = agente.hill_climbing()
entorno.imprimir_tablero(nodo_hc.tablero)
print(f"H Óptimo: {nodo_hc.h}")
print(f"Iteraciones: {iter_hc}")

print("\nRESULTADO SIMULATED ANNEALING (1 Corrida)")
nodo_sa, iter_sa = agente.simulated_annealing(
    alpha=0.99
)  # Dato estandar para este espacio de busqueda, con ese nivel de intentos
entorno.imprimir_tablero(nodo_sa.tablero)
print(f"H Óptimo: {nodo_sa.h}")
print(f"Iteraciones: {iter_sa}")

# 2. Datos Estadísticos (30 Corridas Silenciosas)
print("\n" + "=" * 45)
print(" DATOS ESTADÍSTICOS (30 Ejecuciones)")
print("=" * 45)

corridas = 30

exitos_hc = sum(1 for _ in range(corridas) if agente.hill_climbing()[0].h == 0)
exitos_sa = sum(1 for _ in range(corridas) if agente.simulated_annealing()[0].h == 0)

print(f"Tasa de éxito HILL CLIMBING       : {(exitos_hc / corridas) * 100:.1f} %")
print(f"Tasa de éxito SIMULATED ANNEALING : {(exitos_sa / corridas) * 100:.1f} %")
print("=" * 45)


# =========================================
# SUDOKU 9x9

entorno = EntornoSudoku(sudoku_9x9)
agente = AgenteLocal(entorno)

print("ESTADO INICIAL 9x9")
entorno.imprimir_tablero(sudoku_9x9)

# 1. Demostración Visual
print("\nRESULTADO HILL CLIMBING (1 Corrida)")
nodo_hc, iter_hc = agente.hill_climbing()
entorno.imprimir_tablero(nodo_hc.tablero)
print(f"H Óptimo: {nodo_hc.h}")
print(f"Iteraciones: {iter_hc}")

print("\nRESULTADO SIMULATED ANNEALING (1 Corrida)")
nodo_sa, iter_sa = agente.simulated_annealing(
    temp_inicial=100.0, alpha=0.9999, max_iteraciones=500000
)  # Mas intentos para explorar ya que el espacio 9x9 es mas propenso a no encontrarse
entorno.imprimir_tablero(nodo_sa.tablero)
print(f"H Óptimo: {nodo_sa.h}")
print(f"Iteraciones: {iter_sa}")

# 2. Datos Estadísticos (30 Corridas Silenciosas)
print("\n" + "=" * 45)
print(" DATOS ESTADÍSTICOS (30 Ejecuciones)")
print("=" * 45)

corridas = 30

exitos_hc = sum(1 for _ in range(corridas) if agente.hill_climbing()[0].h == 0)
exitos_sa = sum(1 for _ in range(corridas) if agente.simulated_annealing()[0].h == 0)

print(f"Tasa de éxito HILL CLIMBING       : {(exitos_hc / corridas) * 100:.1f} %")
print(f"Tasa de éxito SIMULATED ANNEALING : {(exitos_sa / corridas) * 100:.1f} %")
print("=" * 45)
