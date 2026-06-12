# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Configuración del entorno

# MODIFICAR ACÁ: True para modo Estocástico, False para modo Determinístico
MODO_ESTOCASTICO = True

env = gym.make('FrozenLake-v1', is_slippery=MODO_ESTOCASTICO, render_mode=None)


# Implementación de Q-Learning (Pseudocódigo Slide 90)
def q_learning(env, episodios=10000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999):
    cant_estados = env.observation_space.n
    cant_acciones = env.action_space.n
    
    Q = np.zeros((cant_estados, cant_acciones))
    recompensas_por_episodio = []
    
    print(f"Iniciando entrenamiento de Q-Learning ({episodios} episodios)...")
    
    for ep in range(episodios):
        epsilon = max(epsilon * epsilon_decay, 0.01)
        
        observacion, info = env.reset()
        estado = int(observacion) if isinstance(observacion, (int, np.integer)) else int(observacion[0])
        
        episodio_terminado = False
        recompensa_acumulada = 0
        
        while not episodio_terminado:
            if random.uniform(0, 1) < epsilon:
                accion = env.action_space.sample()  
            else:
                accion = np.argmax(Q[estado, :])
                
            sig_obs, recompensa, terminado, truncado, info = env.step(accion)
            sig_estado = int(sig_obs) if isinstance(sig_obs, (int, np.integer)) else int(sig_obs[0])
            
            episodio_terminado = terminado or truncado
            
            mejor_sig_accion = np.argmax(Q[sig_estado, :])
            objetivo_td = recompensa + gamma * Q[sig_estado, mejor_sig_accion]
            error_td = objetivo_td - Q[estado, accion]
            Q[estado, accion] += alpha * error_td
            
            estado = sig_estado
            recompensa_acumulada += recompensa
            
        recompensas_por_episodio.append(recompensa_acumulada)
        
    # Extraer la política final
    politica = np.argmax(Q, axis=1)
    return Q, politica, recompensas_por_episodio

# Gráfico de la política y valores Q (Mapa de Calor)
def graficar_politica(valores, politica, env):
    desc = env.unwrapped.desc.astype('U')
    shape = desc.shape
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Crear un mapa de colores RGB
    color_map = np.zeros((shape[0], shape[1], 3))
    
    # Normalizar valores para el degradado
    v_max = np.max(valores) if np.max(valores) > 0 else 1.0
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            estado = i * shape[1] + j
            if desc[i, j] == 'H':
                color_map[i, j] = [0.7, 0.0, 0.0] # Rojo oscuro para hoyos
            elif desc[i, j] == 'G':
                color_map[i, j] = [0.0, 0.8, 0.0] # Verde brillante para meta
            else:
                # Degradado de verde según el valor
                v_norm = valores[estado] / v_max
                # Evitar que quede totalmente negro, base = 0.2
                color_map[i, j] = [0.0, 0.2 + 0.6 * v_norm, 0.0]
                
    ax.imshow(color_map)
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            estado = i * shape[1] + j
            
            if desc[i, j] == 'H':
                ax.text(j, i, "0.0", ha='center', va='center', fontsize=20, fontweight='bold', color='white')
            elif desc[i, j] == 'G':
                ax.text(j, i, "G", ha='center', va='center', fontsize=24, fontweight='bold', color='white')
            else:
                # Mostrar el valor en el centro
                val_str = f"{valores[estado]:.2f}"
                ax.text(j, i, val_str, ha='center', va='center', fontsize=16, fontweight='bold', color='white')
                
                # Mostrar el triángulo de la acción en el borde
                accion = politica[estado]
                if accion == 0: # Izquierda
                    ax.text(j - 0.35, i, '◄', ha='center', va='center', fontsize=14, color='white')
                elif accion == 1: # Abajo
                    ax.text(j, i + 0.35, '▼', ha='center', va='center', fontsize=14, color='white')
                elif accion == 2: # Derecha
                    ax.text(j + 0.35, i, '►', ha='center', va='center', fontsize=14, color='white')
                elif accion == 3: # Arriba
                    ax.text(j, i - 0.35, '▲', ha='center', va='center', fontsize=14, color='white')
            
    ax.set_xticks(np.arange(shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Mapa de Calor: Valores Q y Política", pad=20, fontsize=16)
    plt.show()


# Ejecución del entrenamiento
Q_tabla, politica_q, historial_recompensas = q_learning(env, episodios=10000)
print("\n¡Entrenamiento finalizado!")
print(f"Tabla Q resultante:\n{Q_tabla}")
print(f"Política derivada de Q:\n{politica_q}")

# Evaluación de Tasa de Éxito en 1000 episodios
episodios_eval = 1000
victorias = 0
env_eval = gym.make('FrozenLake-v1', is_slippery=MODO_ESTOCASTICO, render_mode=None)

print(f"\nEvaluando tasa de éxito en {episodios_eval} episodios (Modo estocástico: {MODO_ESTOCASTICO})...")
for ep in range(episodios_eval):
    obs, info = env_eval.reset()
    terminado = False
    
    while not terminado:
        estado_eval = int(obs) if isinstance(obs, (int, np.integer)) else int(obs[0])
        accion = int(politica_q[estado_eval])
        obs, recompensa, done, trunc, info = env_eval.step(accion)
        terminado = done or trunc
        
        if done and recompensa == 1.0:
            victorias += 1

env_eval.close()
tasa_exito = (victorias / episodios_eval) * 100
print(f"Tasa de éxito de la política aprendida: {tasa_exito:.2f}% ({victorias} victorias)")

# Fase de prueba (1 episodio)
print("\nEjecutando episodio de prueba final...")
observacion, info = env.reset()
episodio_terminado = False
recompensa_total = 0

while not episodio_terminado:
    estado_actual = int(observacion) if isinstance(observacion, (int, np.integer)) else int(observacion[0])
    accion = int(politica_q[estado_actual]) # Acción inteligente derivada de Q
    
    observacion, recompensa, terminado, truncado, info = env.step(accion)
    recompensa_total += recompensa
    episodio_terminado = terminado or truncado

print(f"Episodio de prueba finalizado. Recompensa: {recompensa_total}")
env.close()

# Generación de gráfica

plt.figure(figsize=(10, 5))

window = 200
medias_moviles = np.convolve(historial_recompensas, np.ones(window)/window, mode='valid')

plt.plot(historial_recompensas, alpha=0.15, color='blue', label='Recompensa por episodio')
plt.plot(medias_moviles, color='red', linewidth=2, label=f'Media móvil (ventana {window})')
plt.title(f"Rendimiento de Q-Learning")
plt.xlabel("Episodios")
plt.ylabel("Recompensa")
plt.legend()
plt.grid(True)

# Guarda la imagen automáticamente en tu carpeta
nombre_grafico = "q_learning_estocastico.png" if MODO_ESTOCASTICO else "q_learning_deterministico.png"
plt.savefig(nombre_grafico, dpi=300)
print(f"Gráfica guardada con éxito como: {nombre_grafico}")
plt.show()

# Calcular el valor del estado a partir de Q (el valor máximo esperado)
valores_q = np.max(Q_tabla, axis=1)
graficar_politica(valores_q, politica_q, env)
