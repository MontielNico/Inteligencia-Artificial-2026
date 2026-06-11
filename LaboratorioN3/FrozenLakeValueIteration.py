# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# Implementación de value iteration
def value_iteration(env, gamma=0.99, theta=1e-8):
    cant_estados = env.observation_space.n
    cant_acciones = env.action_space.n

    V = np.zeros(cant_estados)
    politica = np.zeros(cant_estados, dtype=int)

    while True:
        delta = 0
        for s in range(cant_estados):
            v_old = V[s]
            q_values = []

            for a in range(cant_acciones):
                q_sa = 0

                for prob, sig_estado, recompensa, terminado in env.unwrapped.P[s][a]:
                    q_sa += prob * (recompensa + gamma * V[sig_estado])
                q_values.append(q_sa)
            
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
        
        if delta < theta:
            break
    
    # Extraer política
    for s in range(cant_estados):
        q_values = []
        for a in range(cant_acciones):
            q_sa = 0
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                q_sa += prob * (reward + gamma * V[next_state])
            q_values.append(q_sa)
        politica[s] = np.argmax(q_values)

    return V, politica

# Gráfico de la política y valores (Mapa de Calor)
def graficar_politica(valores, politica, env):
    desc = env.unwrapped.desc.astype('U')
    shape = desc.shape
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Crear un mapa de colores RGB
    color_map = np.zeros((shape[0], shape[1], 3))
    
    # Normalizar valores para el degradado (ignorando posibles valores negativos en FrozenLake no los hay)
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
    plt.title("Mapa de Calor: Valores y Política Óptima", pad=20, fontsize=16)
    plt.show()


# Configuración del entorno y cálculo
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")

valores_optimos, politica_optima = value_iteration(env)

print("Valores Optimos:", valores_optimos)
print("Politica Optima:", politica_optima)

# Graficar la política óptima y el mapa de calor
graficar_politica(valores_optimos, politica_optima, env)

# Ejecución con politica óptima
observacion, info = env.reset()

episodio_terminado = False
recompensa_total = 0

while not episodio_terminado:
    
    accion = int(politica_optima[observacion])
    observacion, recompensa, episodio_terminado, truncamiento, info = env.step(accion)
    recompensa_total += recompensa
    episodio_terminado = episodio_terminado or truncamiento

print(f"Recompensa obtenida: {recompensa_total}")
env.close()