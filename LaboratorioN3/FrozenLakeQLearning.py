# pyrefly: ignore [missing-import]
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt

# Configuración del entorno

# MODIFICAR ACÁ: True para modo Estocástico, False para modo Determinístico
MODO_ESTOCASTICO = False 

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

# Gráfico de la política óptima
def graficar_politica(politica, env):
    desc = env.unwrapped.desc.astype('U')
    shape = desc.shape
    
    acciones_str = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Mapa de colores para las casillas
    color_map = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if desc[i, j] == 'S': color_map[i, j] = 0.5
            elif desc[i, j] == 'F': color_map[i, j] = 0.8
            elif desc[i, j] == 'H': color_map[i, j] = 0.0
            elif desc[i, j] == 'G': color_map[i, j] = 1.0
            
    ax.matshow(color_map, cmap='coolwarm')
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            estado = i * shape[1] + j
            if desc[i, j] in ['H', 'G']:
                ax.text(j, i, desc[i, j], ha='center', va='center', fontsize=20, fontweight='bold', color='white' if desc[i, j] == 'H' else 'black')
            else:
                accion = politica[estado]
                ax.text(j, i, acciones_str[accion], ha='center', va='center', fontsize=24)
            
    ax.set_xticks(np.arange(shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Política Óptima (Gris: Inicio, Naranja: Hielo, H: Hoyo, G: Meta)", pad=20)
    plt.show()


# Ejecución del entrenamiento
Q_tabla, politica_q, historial_recompensas = q_learning(env, episodios=10000)
print("\n¡Entrenamiento finalizado!")
print(f"Tabla Q resultante:\n{Q_tabla}")
print(f"Política derivada de Q:\n{politica_q}")

# Fase de prueba
print("\nEjecutando episodio de prueba con la política aprendida...")
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
plt.title(f"Rendimiento de Q-Learning - ¿Es estocástico?: {MODO_ESTOCASTICO}")
plt.xlabel("Episodios")
plt.ylabel("Recompensa")
plt.legend()
plt.grid(True)

# Guarda la imagen automáticamente en tu carpeta
nombre_grafico = "q_learning_estocastico.png" if MODO_ESTOCASTICO else "q_learning_deterministico.png"
plt.savefig(nombre_grafico, dpi=300)
print(f"Gráfica guardada con éxito como: {nombre_grafico}")
plt.show()
graficar_politica(politica_q, env)
