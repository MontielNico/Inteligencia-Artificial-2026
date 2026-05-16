import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import math
from deap import base, creator, tools, algorithms

# ==========================================
# CONSTANTES DEL PROBLEMA
# ==========================================


def potenciaBase(p, Cv, R, V):
    """Función para el calculo de Potencia Base de un Aerogenerador"""

    area = math.pi * (R**2)
    return 0.5 * p * area * Cv * (V**3)


GRID_SIZE = 20  # El terreno es una cuadrícula de 20x20 celdas discretas
N_MOLINOS = 25  # Número fijo de aerogeneradores a colocar
BETA = 0.08  # 8% de reducción por cada estela
P_BASE = potenciaBase(1.225, 0.40, 40, 12)  # MW por molino
# Potencia base un aerogenerador sin estelas: 2.1280394653180403 Mw
# Potencia maxima teorica (25 molinos sin estelas): 53.20098663295101 Mw
# Todos estos datos recuparados de la siguiente ecuación: P = 0.5 * p * A * Cv * V^3
#   ρ = 1,225 kg/m3 (Densidad del aire)
#   Cp = 0,40 (Coeficiente de potencia)
#   R = 40 m (radio del rotor)
#   A = πR2 (área generada por el rotor)
#   v = 12 m/s (velocidad del viento)

# MAXIMO_TEORICO = (P_BASE * N_MOLINOS) / 1000000

# ==========================================
# CONFIGURACIÓN DE DEAP (POBLACIÓN, TOOLBOX)
# ==========================================

# Queremos MAXIMIZAR la energía, por lo que el peso es positivo (1.0)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individuo", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def generar_coordenada():
    """Función generadora de coordenadas aleatorias dentro del terreno."""

    return (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))


def generar_individuo():
    """Función generadora de individuos (conjunto de coordenadas únicas)."""

    molinos = set()
    while len(molinos) < N_MOLINOS:
        molinos.add(generar_coordenada())
    return creator.Individuo(list(molinos))


toolbox.register("individuo", generar_individuo)
toolbox.register("poblacion", tools.initRepeat, list, toolbox.individuo)

# ==========================================
# FUNCIONES DE FITNESS, CRUCE Y MUTACIÓN
# ==========================================


def calculo_fitness(individuo):
    """Calcula la energía total producida por el parque considerando las estelas. Función de FITNESS."""

    potenciaBas = potenciaBase(1.225, 0.4, 40, 12)
    potenciaTotal = 0.0
    set_individuo = set(individuo)

    # Penalización por no alcanzar los 25 aerogeneradores solicitado, es por eso que lo aleja ante posibilidad de competir con demas individuos
    if len(set_individuo) != 25:
        return (0.0,)

    for f, c in individuo:
        penalizaciones = 0

        # Revisamos las 3 celdas al norte
        for i in range(1, 4):
            if (f - i, c) in set_individuo:
                penalizaciones += 1

        # Revisamos las 3 celdas al este
        for i in range(1, 4):
            if (f, c + i) in set_individuo:
                penalizaciones += 1

        potenciaAerogenerador = potenciaBas * max(0, 1 - (0.08 * penalizaciones))
        potenciaTotal += potenciaAerogenerador

    potenciaTotal /= 1000000
    return (potenciaTotal,)


def cruce_un_punto(ind1, ind2):
    """Cruza dos individuos asegurando que no haya molinos duplicados. Cruce unico PUNTO"""

    hijo1Set = set(ind1[:12] + ind2[12:])
    hijo2Set = set(ind2[:12] + ind1[12:])

    hijo1 = list(hijo1Set)[:N_MOLINOS]
    hijo2 = list(hijo2Set)[:N_MOLINOS]

    # Si faltan molinos (porque se borraron duplicados), agregamos aleatorios
    while len(hijo1) < N_MOLINOS:
        nuevo = generar_coordenada()
        if nuevo not in hijo1:
            hijo1.append(nuevo)

    while len(hijo2) < N_MOLINOS:
        nuevo = generar_coordenada()
        if nuevo not in hijo2:
            hijo2.append(nuevo)

    for i in range(N_MOLINOS):
        ind1[i], ind2[i] = hijo1[i], hijo2[i]

    return ind1, ind2


def cruce_dos_puntos(ind1, ind2):
    """Cruza dos individuos asegurando que no haya molinos duplicados. Cruce doble PUNTOS"""

    hijo1Set = set(ind1[:8] + ind2[8:16] + ind1[16:])
    hijo2Set = set(ind2[:8] + ind1[8:16] + ind2[16:])

    hijo1 = list(hijo1Set)[:N_MOLINOS]
    hijo2 = list(hijo2Set)[:N_MOLINOS]

    # Si faltan molinos (porque se borraron duplicados), agregamos aleatorios
    while len(hijo1) < N_MOLINOS:
        nuevo = generar_coordenada()
        if nuevo not in hijo1:
            hijo1.append(nuevo)

    while len(hijo2) < N_MOLINOS:
        nuevo = generar_coordenada()
        if nuevo not in hijo2:
            hijo2.append(nuevo)

    for i in range(N_MOLINOS):
        ind1[i], ind2[i] = hijo1[i], hijo2[i]

    return ind1, ind2


def mutacion_parque(individuo, probMutacion):
    """Muta moviendo aleatoriamente algunos molinos a casillas vacías."""

    for i in range(N_MOLINOS):
        if random.random() < probMutacion:  # Probabilidad de mutación de una coordenada
            nuevo_f, nuevo_c = generar_coordenada()

            while (nuevo_f, nuevo_c) in individuo:
                nuevo_f, nuevo_c = generar_coordenada()
            individuo[i] = (nuevo_f, nuevo_c)

    return (individuo,)


def mutacion_deslizamiento(individuo, prob_mutacion_gen):
    """
    Mutacion 'empujando' un molino 1 casilla hacia una celda adyacente vacía.
    """
    movimientos = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    for i in range(N_MOLINOS):
        if (  # Probabilidad de mutación de una coordenada
            random.random() < prob_mutacion_gen
        ):
            f_actual, c_actual = individuo[i]

            random.shuffle(movimientos)  # Aleatoridad para el movimiento dado

            for df, dc in movimientos:
                nuevo_f = f_actual + df
                nuevo_c = c_actual + dc

                if 0 <= nuevo_f < GRID_SIZE and 0 <= nuevo_c < GRID_SIZE:
                    # Para no caerse de la cuadrilla
                    if (nuevo_f, nuevo_c) not in individuo:
                        individuo[i] = (nuevo_f, nuevo_c)
                        break

    return (individuo,)


# ==========================================
# REGISTRAMOS LAS FUNCIONES EN EL TOOLBOX DE DEAP
# ==========================================

toolbox.register("evaluate", calculo_fitness)
toolbox.register("mate", cruce_un_punto)
# cruce_un_punto / cruce_dos_puntos
toolbox.register("mutate", mutacion_parque, probMutacion=0.1)
# Mutacion_normal / #mutacion_desplazamiento
# Probabilidad basada en el individuo mismo, ya que es el individuo seleccionado
toolbox.register("select", tools.selTournament, tournsize=3)


# ==========================================
# BUCLE PRINCIPAL (EL MOTOR EVOLUTIVO)
# ==========================================
def ejecutar_una_vez(verbose=False):
    # Parámetros del experimento
    TAMANO_POBLACION = 30
    GENERACIONES = 30
    PROB_CRUCE = 0.7
    PROB_MUTACION = 0.3

    poblacion = toolbox.poblacion(n=TAMANO_POBLACION)
    salon_fama = tools.HallOfFame(1)  # Guardará el mejor de todos los tiempos

    # Estadísticas con NumPy
    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0])
    estadisticas.register("Promedio", np.mean)
    estadisticas.register("Maximo", np.max)

    print("Iniciando Evolución del Parque Eólico...")

    # Usamos MuPlusLambda (Padres compiten con Hijos) para garantizar Elitismo
    poblacion, logbook = algorithms.eaMuPlusLambda(
        population=poblacion,
        toolbox=toolbox,
        mu=TAMANO_POBLACION,
        lambda_=TAMANO_POBLACION,
        cxpb=PROB_CRUCE,
        mutpb=PROB_MUTACION,
        ngen=GENERACIONES,
        stats=estadisticas,
        halloffame=salon_fama,
        verbose=True,
    )

    campeon = salon_fama[0]
    energia_maxima = campeon.fitness.values[0]

    print("=" * 40)
    print("🥇 ALGORITMO GENETICO TERMINADO")
    print(f"Mejor Energía Obtenida: {energia_maxima:.2f})")
    print("=" * 40)

    return logbook, campeon


# ==========================================
# 6. EXPERIMENTO: N_RUNS EJECUCIONES
# ==========================================
def experimento_multiple(n_runs=30):
    """
    Corre el AG n_runs veces independientes.
    Devuelve:
      - all_maximos:  matriz (n_runs x generaciones+1) con el mejor fitness por gen
      - all_promedios: ídem para el promedio
      - mejores_finales: lista con el mejor fitness de cada ejecución
      - mejor_campeon: el mejor individuo de todas las ejecuciones
    """
    print(f"Iniciando {n_runs} ejecuciones independientes del AG...")
    all_maximos = []
    all_promedios = []
    mejores_finales = []
    mejor_campeon = None
    mejor_fitness_global = -np.inf

    for run in range(1, n_runs + 1):
        logbook, campeon = ejecutar_una_vez(verbose=False)
        maximos = logbook.select("Maximo")
        promedios = logbook.select("Promedio")

        all_maximos.append(maximos)
        all_promedios.append(promedios)
        fitness_final = campeon.fitness.values[0]
        mejores_finales.append(fitness_final)

        if fitness_final > mejor_fitness_global:
            mejor_fitness_global = fitness_final
            mejor_campeon = campeon

        print(f"  Ejecución {run:02d}/{n_runs} — Mejor final: {fitness_final:.2f} MW")

    all_maximos = np.array(all_maximos)  # shape: (n_runs, generaciones+1)
    all_promedios = np.array(all_promedios)

    print("=" * 50)
    print(" ✅ Experimento terminado.")
    print(f"   Mejor absoluto : {np.max(mejores_finales):.2f} MW")
    print(f"   Media final    : {np.mean(mejores_finales):.2f} MW")
    print(f"   Desv. estándar : {np.std(mejores_finales):.2f} MW")
    print(f"   Mínimo final   : {np.min(mejores_finales):.2f} MW")
    print("=" * 50)

    return all_maximos, all_promedios, mejores_finales, mejor_campeon


# ==========================================
# 7. VISUALIZACIÓN
# ==========================================
def graficar_convergencia_estadistica(all_maximos, all_promedios, n_runs):
    """
    Gráfico de convergencia con banda media ± std para el mejor y el promedio.
    """
    generaciones = np.arange(all_maximos.shape[1])

    media_max = np.mean(all_maximos, axis=0)
    std_max = np.std(all_maximos, axis=0)
    media_prom = np.mean(all_promedios, axis=0)
    std_prom = np.std(all_promedios, axis=0)

    fig, ax = plt.subplots(figsize=(11, 6))

    # --- Mejor fitness por generación ---
    ax.plot(
        generaciones,
        media_max,
        color="green",
        linewidth=2.5,
        label=f"Media del Mejor Fitness ({n_runs} runs)",
    )
    ax.fill_between(
        generaciones,
        media_max - std_max,
        media_max + std_max,
        alpha=0.25,
        color="green",
        label="Banda ± 1 Desv. Estándar (Mejor)",
    )

    # --- Promedio por generación ---
    ax.plot(
        generaciones,
        media_prom,
        color="orange",
        linewidth=2,
        linestyle="--",
        label=f"Media del Fitness Promedio ({n_runs} runs)",
    )
    ax.fill_between(
        generaciones,
        media_prom - std_prom,
        media_prom + std_prom,
        alpha=0.20,
        color="orange",
        label="Banda ± 1 Desv. Estándar (Promedio)",
    )

    # --- Referencia teórica ---
    ax.axhline(
        y=53,
        color="red",
        linestyle=":",
        linewidth=1.5,

    )

    ax.set_title(
        f"Curva de Convergencia Estadística — {n_runs} Ejecuciones Independientes\n"
        "Parque Eólico — Algoritmo Genético (eaMuPlusLambda)",
        fontsize=13,
    )
    ax.set_xlabel("Generación", fontsize=12)
    ax.set_ylabel("Producción de Energía (MW)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("3_Convergencia_Estadistica.png", dpi=300)
    plt.show()
    print("Gráfico guardado: 3_Convergencia_Estadistica.png")


def graficar_boxplot_finales(mejores_finales):
    """Boxplot de los mejores fitness finales de cada ejecución."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        mejores_finales,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        medianprops=dict(color="darkgreen", linewidth=2),
    )
    ax.axhline(y=53.25, color="red", linestyle=":", label="Máximo Teórico")
    ax.set_title(
        "Distribución de Mejores Soluciones Finales\n(30 ejecuciones independientes)"
    )
    ax.set_ylabel("Producción de Energía (MW)")
    ax.set_xticks([1])
    ax.set_xticklabels(["AG Parque Eólico"])
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("4_Boxplot_Finales.png", dpi=300)
    plt.show()
    print("Gráfico guardado: 4_Boxplot_Finales.png")


def graficar_mapa_campeon(mejor_campeon):
    matriz_terreno = np.zeros((GRID_SIZE, GRID_SIZE))
    for x, y in mejor_campeon:
        matriz_terreno[y][x] = 1

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        matriz_terreno, cmap="YlGn", cbar=False, linewidths=0.5, linecolor="gray"
    )

    sprite = plt.imread(r"images/molino.png")

    for x, y in mejor_campeon:
        plt.imshow(sprite, extent=[x + 0.1, x + 0.9, y + 0.9, y + 0.1], zorder=10)

    plt.title(
        f"Disposición Óptima del Parque Eólico\n"
        f"Producción: {mejor_campeon.fitness.values[0]:.2f} MW"
    )
    plt.xlabel("Oeste → Este")
    plt.ylabel("Norte → Sur")
    plt.savefig("2_Mapa_Parque_Eolico.png", dpi=300)
    plt.show()


# ==========================================
# PUNTO DE ENTRADA
# ==========================================
if __name__ == "__main__":
    N_RUNS = 30  # Valor para las ejecuciones

    # Correr experimento
    all_maximos, all_promedios, mejores_finales, mejor_campeon = experimento_multiple(
        N_RUNS
    )

    # Gráfico 1: Convergencia con banda estadística (el que pedía el proyecto)
    graficar_convergencia_estadistica(all_maximos, all_promedios, N_RUNS)

    # Gráfico 2: Mapa del mejor parque encontrado
    graficar_mapa_campeon(mejor_campeon)

    # Gráfico 3: Boxplot de resultados finales (análisis estadístico)
    graficar_boxplot_finales(mejores_finales)

    # Resumen estadístico en consola
    print("\n📊 RESUMEN ESTADÍSTICO COMPLETO:")
    print(f"   N ejecuciones  : {N_RUNS}")
    print(f"   Media          : {np.mean(mejores_finales):.4f} MW")
    print(f"   Desv. Estándar : {np.std(mejores_finales):.4f} MW")
    print(f"   Mínimo         : {np.min(mejores_finales):.4f} MW")
    print(f"   Máximo         : {np.max(mejores_finales):.4f} MW")
    print(f"   Mediana        : {np.median(mejores_finales):.4f} MW")
