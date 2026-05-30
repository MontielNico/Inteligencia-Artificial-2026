from deap import benchmarks
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import math
from deap import base, creator, tools, algorithms
import time
import scipy.stats as stats

# ==========================================
# CONSTANTES DEL PROBLEMA
# ==========================================
def potenciaBase(p, Cv, R, V):
    """Función para el calculo de Potencia Base de un Aerogenerador"""

    area = math.pi * (R**2)
    return 0.5 * p * area * Cv * (V**3)

GRID_SIZE = 20 
N_MOLINOS = 25  
BETA = 0.08  
P_BASE = potenciaBase(1.225, 0.40, 40, 12) 

# Parámetros estándar del Algoritmo Genético
POP_SIZE_ESTANDAR = 20
CX_PB_ESTANDAR = 0.50
MUT_PB_ESTANDAR = 0.05
N_GEN_ESTANDAR = 40
N_RUNS_ESTANDAR = 30

# ==========================================
# CONFIGURACIÓN DE DEAP (POBLACIÓN, TOOLBOX)
# ==========================================

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

    potenciaBas = P_BASE
    potenciaTotal = 0.0
    set_individuo = set(individuo)

    
    if len(set_individuo) != N_MOLINOS:
        # Penalización por no alcanzar los 25 aerogeneradores solicitado, es por eso que lo aleja ante posibilidad de competir con demas individuos
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
toolbox.register("mutate", mutacion_parque, probMutacion=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# ==========================================
# MOTOR EVOLUTIVO PARAMETRIZADO
# ==========================================
def ejecutar_una_vez(pop_size, cx_pb, mut_pb, n_gen=30, verbose=False):
    """Ejecuta una vez el Algoritmo Genético con parámetros específicos."""
    poblacion = toolbox.poblacion(n=pop_size)
    salon_fama = tools.HallOfFame(1)

    estadisticas = tools.Statistics(lambda ind: ind.fitness.values[0])
    estadisticas.register("Maximo", np.max)

    # eaMuPlusLambda garantiza elitismo (padres e hijos compiten por sobrevivir)
    poblacion, logbook = algorithms.eaMuPlusLambda(
        population=poblacion,
        toolbox=toolbox,
        mu=pop_size,
        lambda_=pop_size,
        cxpb=cx_pb,
        mutpb=mut_pb,
        ngen=n_gen,
        stats=estadisticas,
        halloffame=salon_fama,
        verbose=verbose,
    )
    return logbook, salon_fama[0]

def correr_analisis_comparativo():
    """
    Orquestador que corre múltiples ejecuciones para distintos escenarios
    y genera gráficos comparativos y una tabla resumen.
    """
    estandar = [POP_SIZE_ESTANDAR, CX_PB_ESTANDAR, MUT_PB_ESTANDAR]
    alta_poblacion = 60
    alta_cruce = 0.8
    alta_mutacion = 0.3
    escenarios = [
        {"nombre": f"Estándar ({estandar[0]}, {estandar[1]}, {estandar[2]})", "pop": estandar[0], "cx": estandar[1], "mut": estandar[2]},
        {"nombre": f"Alta Cruce ({estandar[0]}, {alta_cruce}, {estandar[2]})", "pop": estandar[0], "cx": alta_cruce, "mut": estandar[2]},
        {"nombre": f"Alta Población ({alta_poblacion}, {estandar[1]}, {estandar[2]})", "pop": alta_poblacion, "cx": estandar[1], "mut": estandar[2]},
        {"nombre": f"Alta Mutación ({estandar[0]}, {estandar[1]}, {alta_mutacion})", "pop": estandar[0], "cx": estandar[1], "mut": alta_mutacion},
    ] 

    N_RUNS = N_RUNS_ESTANDAR  # Ejecuciones por cada escenario para validez estadística
    N_GEN = N_GEN_ESTANDAR  # Generaciones por cada ejecución
    resumen_estadistico = []
    todos_los_resultados = {}
    todas_las_convergencias = {}
    
    mejor_campeon_global = None
    mejor_fitness_global = -np.inf

    # Óptimo de referencia para el Hit Rate (Teórico sin estelas)
    OPTIMO_REF = (N_MOLINOS * P_BASE) / 1000000

    print(f"Iniciando Benchmarking: {len(escenarios)} escenarios x {N_RUNS} runs...")

    for esc in escenarios:
        print(f"\n Analizando escenario: {esc['nombre']}...")
        finales_escenario = []
        convergencias_escenario = []

        start_time_escenario = time.time()

        for r in range(1, N_RUNS + 1):
            log, campeon = ejecutar_una_vez(esc["pop"], esc["cx"], esc["mut"], N_GEN)
            fitness_actual = campeon.fitness.values[0]
            finales_escenario.append(fitness_actual)
            convergencias_escenario.append(log.select("Maximo"))
            
            # Evaluar y guardar si es el mejor campeón de TODOS los escenarios
            if fitness_actual > mejor_fitness_global:
                mejor_fitness_global = fitness_actual
                mejor_campeon_global = campeon
                    
            if r % 10 == 0:
                print(f"  Progreso: {r}/{N_RUNS} ejecuciones completadas.")

        tiempo_total = time.time() - start_time_escenario

        # --- Cálculo de Métricas ---
        mejor = np.max(finales_escenario)
        media = np.mean(finales_escenario)
        std = np.std(finales_escenario)

        # Hit Rate: % de veces que llega al óptimo (con margen de error de 0.01)
        hits = sum(1 for f in finales_escenario if f >= (OPTIMO_REF - 0.01))
        hit_rate = (hits / N_RUNS) * 100

        # Confiabilidad: Proporción entre media y mejor resultado
        confiabilidad = (media / mejor * 100) if mejor > 0 else 0

        resumen_estadistico.append(
            {
                "Config": esc["nombre"],
                "Mejor": mejor,
                "Media": media,
                "Std": std,
                "Hit Rate": hit_rate,
                "Confiabilidad": confiabilidad,
                "T. Total": tiempo_total,
            }
        )

        todos_los_resultados[esc["nombre"]] = finales_escenario
        todas_las_convergencias[esc["nombre"]] = np.mean(
            convergencias_escenario, axis=0
        )

    # Imprimir Tabla y generar gráficos
    imprimir_tabla_resumen(resumen_estadistico)
    graficar_resultados_comparativos(todos_los_resultados, todas_las_convergencias, N_RUNS)

    if mejor_campeon_global is not None:
        graficar_mapa_campeon(mejor_campeon_global)

    # Realizar análisis de normalidad (KS)
    realizar_test_normalidad_ks(todos_los_resultados)

    # Realizar análisis comparativo de distribuciones (Kruskal-Wallis)
    realizar_test_kruskal_wallis(todos_los_resultados)

    # Realizar análisis comparativo por parejas (Wilcoxon)
    realizar_test_wilcoxon_parejas(todos_los_resultados)


def realizar_test_normalidad_ks(todos_los_resultados):
    """
    Aplica el test de normalidad de Kolmogorov-Smirnov sobre los datos en memoria,
    imprimiendo en consola si son o no normales de forma directa.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISIS DE NORMALIDAD (TEST DE KOLMOGOROV-SMIRNOV)")
    print("=" * 80)
    
    alpha = 0.05
    for nombre, valores in todos_los_resultados.items():
        arr_valores = np.array(valores)
        media = np.mean(arr_valores)
        desv = np.std(arr_valores, ddof=1)
        
        print(f"Escenario: {nombre}")
        
        # Caso de varianza cero
        if desv == 0:
            print("  Resultado: NO ES NORMAL (Varianza nula, todos los valores son idénticos)")
            print("-" * 80)
            continue
            
        # Test Kolmogorov-Smirnov contra una distribución normal de misma media y desviación estándar
        stat, p_val = stats.kstest(arr_valores, 'norm', args=(media, desv))
        
        es_normal = p_val >= alpha
        resultado_str = "SÍ ES NORMAL (Éxito)" if es_normal else "NO ES NORMAL (Fallo)"
        
        print(f"  Estadístico KS: {stat:.5f} | p-valor: {p_val:.5e}")
        print(f"  Resultado: {resultado_str}")
        print("-" * 80)
    print("=" * 80 + "\n")


def realizar_test_kruskal_wallis(todos_los_resultados):
    """
    Aplica el test no paramétrico de Kruskal-Wallis sobre las distribuciones 
    de los 4 escenarios para verificar si son estadísticamente iguales entre sí.
    """
    print("\n" + "=" * 80)
    print(" ANÁLISIS COMPARATIVO DE DISTRIBUCIONES: TEST DE KRUSKAL-WALLIS")
    print("=" * 80)
    print("Hipótesis Nula (H0): Los 4 escenarios tienen distribuciones idénticas (medianas iguales).")
    print("Hipótesis Alternativa (H1): Al menos un escenario difiere significativamente.\n")
    
    alpha = 0.05
    listas_valores = list(todos_los_resultados.values())
    
    # Ejecución del test de Kruskal-Wallis
    stat, p_val = stats.kruskal(*listas_valores)
    
    # Interpretación
    son_iguales = p_val >= alpha
    resultado_str = "SÍ SON IGUALES (No se rechaza H0 - No hay diferencias significativas)" if son_iguales else "NO SON IGUALES (Se rechaza H0 - Hay diferencias significativas)"
    
    print(f"Estadístico H de Kruskal-Wallis: {stat:.5f}")
    print(f"p-valor: {p_val:.5e}")
    print(f"Resultado: {resultado_str}")
    print("=" * 80 + "\n")


def realizar_test_wilcoxon_parejas(todos_los_resultados):
    """
    Realiza el test de rangos con signo de Wilcoxon entre todos los pares posibles 
    de los 4 escenarios para verificar si existen diferencias significativas entre ellas.
    """
    import itertools
    
    print("\n" + "=" * 80)
    print(" ANÁLISIS COMPARATIVO EN PAREJAS: TEST DE WILCOXON")
    print("=" * 80)
    print("Hipótesis Nula (H0): No hay diferencia sistemática entre ambos escenarios.")
    print("Hipótesis Alternativa (H1): Existe una diferencia significativa entre ambos escenarios.")
    print("Nivel de significancia (alpha): 0.05\n")
    
    nombres = list(todos_los_resultados.keys())
    pares = list(itertools.combinations(nombres, 2))
    
    alpha = 0.05
    
    for esc1, esc2 in pares:
        val1 = np.array(todos_los_resultados[esc1])
        val2 = np.array(todos_los_resultados[esc2])
        
        print(f"Comparación: {esc1} vs {esc2}")
        
        # Verificar si todos los elementos son idénticos en diferencia (varianza cero de la diferencia)
        diff = val1 - val2
        if np.all(diff == 0):
            print("  Estadístico Wilcoxon: N/A (Muestras idénticas)")
            print("  p-valor: 1.00000e+00")
            print("  Resultado: SIN DIFERENCIA (Fallo - Muestras idénticas)")
            print("-" * 80)
            continue
            
        try:
            # Test Wilcoxon Signed-Rank
            stat, p_val = stats.wilcoxon(val1, val2)
            
            # Interpretación
            hay_diferencia = p_val < alpha
            resultado_str = "DIFERENCIA DETECTADA (Éxito - Se rechaza H0)" if hay_diferencia else "SIN DIFERENCIA (Fallo - No se rechaza H0)"
            
            print(f"  Estadístico Wilcoxon: {stat:.1f}")
            print(f"  p-valor: {p_val:.5e}")
            print(f"  Resultado: {resultado_str}")
            
        except Exception as e:
            # Control por si falla por otra razón estadística
            print(f"  No se pudo realizar el test Wilcoxon: {e}")
            
        print("-" * 80)
    print("=" * 80 + "\n")





def imprimir_tabla_resumen(datos):

    """Muestra una tabla formateada con los resultados del análisis."""
    print("\n" + "=" * 115)
    print(
        f"{'ESCENARIO':<30} | {'MEJOR (MW)':<10} | {'MEDIA (MW)':<10} | {'STD':<8} | {'HIT%':<7} | {'CONF%':<7} | {'TIEMPO'}"
    )
    print("-" * 115)
    for d in datos:
        print(
            f"{d['Config']:<30} | {d['Mejor']:<10.3f} | {d['Media']:<10.3f} | {d['Std']:<8.3f} | {d['Hit Rate']:<6.1f}% | {d['Confiabilidad']:<6.1f}% | {d['T. Total']:>6.2f}s"
        )
    print("=" * 115 + "\n")


def graficar_resultados_comparativos(resultados, convergencias, n_runs):
    """Genera y guarda los gráficos de la comparativa."""

    # 1. Boxplot Comparativo (Distribución de soluciones finales)
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=list(resultados.values()), palette="Set2")
    plt.xticks(range(len(resultados)), list(resultados.keys()), rotation=15)
    plt.title(
        "Comparativa Estadística: Calidad de Solución Final por Escenario", fontsize=13
    )
    plt.ylabel("Producción de Energía (MW)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("2_Comparativa_Boxplot.png", dpi=300)
    print("Gráfico guardado: 2_Comparativa_Boxplot.png")

    # 2. Curvas de Convergencia (Velocidad de aprendizaje)
    plt.figure(figsize=(12, 7))
    for nombre, curva in convergencias.items():
        plt.plot(curva, label=nombre, linewidth=2.5)
    plt.title(
        f"Comparativa: Velocidad de Convergencia (Media de {n_runs} ejecuciones)", fontsize=13
    )
    plt.xlabel("Generación")
    plt.ylabel("Energía Máxima (MW)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("1_Comparativa_Convergencia.png", dpi=300)
    print("Gráfico guardado: 1_Comparativa_Convergencia.png")

    plt.show()


# ==========================================
# 6. EXPERIMENTO: N_RUNS EJECUCIONES
# ==========================================
def experimento_multiple(pop_size=POP_SIZE_ESTANDAR, cx_pb=CX_PB_ESTANDAR, mut_pb=MUT_PB_ESTANDAR, n_gen=N_GEN_ESTANDAR, n_runs=N_RUNS_ESTANDAR):
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
        logbook, campeon = ejecutar_una_vez(pop_size, cx_pb, mut_pb, n_gen=n_gen, verbose=False)
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
    print(" Experimento terminado.")
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
    plt.savefig("2_Convergencia_Estadistica.png", dpi=300)
    plt.show()
    print("Gráfico guardado: 2_Convergencia_Estadistica.png")


def graficar_boxplot_finales(mejores_finales, n_runs=30):
    """Boxplot de los mejores fitness finales de cada ejecución."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(
        mejores_finales,
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="lightgreen", color="green"),
        medianprops=dict(color="darkgreen", linewidth=2),
    )
    ax.set_title(
        f"Distribución de Mejores Soluciones Finales\n({n_runs} ejecuciones independientes)"
    )
    ax.set_ylabel("Producción de Energía (MW)")
    ax.set_xticks([1])
    ax.set_xticklabels(["AG Parque Eólico"])
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("3_Boxplot_Finales.png", dpi=300)
    plt.show()
    print("Gráfico guardado: 3_Boxplot_Finales.png")


def graficar_mapa_campeon(mejor_campeon):
    matriz_terreno = np.zeros((GRID_SIZE, GRID_SIZE))
    for x, y in mejor_campeon:
        matriz_terreno[y][x] = 1

    plt.figure(figsize=(8, 8))
    sns.heatmap(
        matriz_terreno, cmap="YlGn", cbar=False, linewidths=0.5, linecolor="gray"
    )

    import os
    # Intentamos primero con la ruta absoluta/carpeta superior, luego con ruta local
    if os.path.exists(r"ProyectoAerogeneradores\images\molino.png"):
        sprite = plt.imread(r"ProyectoAerogeneradores\images\molino.png")
        for x, y in mejor_campeon:
            plt.imshow(sprite, extent=[x + 0.1, x + 0.9, y + 0.9, y + 0.1], zorder=10)
    elif os.path.exists(r"images\molino.png"):
        sprite = plt.imread(r"images\molino.png")
        for x, y in mejor_campeon:
            plt.imshow(sprite, extent=[x + 0.1, x + 0.9, y + 0.9, y + 0.1], zorder=10)
    else:
        print("Aviso: No se encontró la imagen del molino. Usando marcadores alternativos.")
        for x, y in mejor_campeon:
            plt.plot(x + 0.5, y + 0.5, marker='*', color='blue', markersize=15, zorder=10)

    plt.title(
        f"Disposición Óptima del Parque Eólico\n"
        f"Producción: {mejor_campeon.fitness.values[0]:.2f} MW"
    )
    plt.xlabel("Oeste → Este")
    plt.ylabel("Norte → Sur")
    plt.savefig("3_Mapa_Parque_Eolico.png", dpi=300)
    plt.show()


# ==========================================
# PUNTO DE ENTRADA
# ==========================================
if __name__ == "__main__":
    # Ejecutamos el análisis comparativo completo
    correr_analisis_comparativo()
