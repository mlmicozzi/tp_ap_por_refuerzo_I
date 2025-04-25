import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict

# --- Parámetros de configuración ---
params = {
    "env_name": "Blackjack-v1",      # Nombre del entorno
    "learning_rate": 0.01,           # Alpha: Tasa de aprendizaje
    "discount_factor": 0.99,         # Gamma: Factor de descuento (valor de recompensas futuras)
    "num_episodes": 200000,          # Número de episodios para entrenar
    "epsilon": 1.0,                  # Tasa de exploración inicial (100% aleatorio)
    "max_epsilon": 1.0,              # Máximo valor de epsilon
    "min_epsilon": 0.01,             # Mínimo valor de epsilon
    "epsilon_decay_rate": 0.0001,    # Tasa de decaimiento de epsilon por episodio
    "report_interval": 10000,        # Intervalo para reportar progreso
    "moving_avg_window": 5000        # Ventana para la media móvil en la gráfica
}

# --- Inicialización del entorno y Q-table ---
env = gym.make(params["env_name"])

# Q-Table como diccionario de diccionarios: estado -> acción -> valor Q
q_table = defaultdict(lambda: np.zeros(env.action_space.n))

# Lista para almacenar recompensas por episodio
rewards_all_episodes = []

# Información inicial
print(f"--- Entrenando con Q-Learning en {params['env_name']} ---")
print(f"Estados: {env.observation_space}, Acciones: {env.action_space}")
print(f"Episodios: {params['num_episodes']}, Alpha={params['learning_rate']}, Gamma={params['discount_factor']}, Epsilon decay={params['epsilon_decay_rate']}")

# --- Entrenamiento ---
start_time = time.time()

for episode in range(params["num_episodes"]):
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Elegir Acción (Epsilon-Greedy)
        if random.uniform(0, 1) < params["epsilon"]:
            action = env.action_space.sample()  # Explorar
        else:
            action = np.argmax(q_table[state])  # Explotar

        # Ejecutar acción
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Asegurarse que el estado siguiente esté en la Q-table
        if next_state not in q_table:
            q_table[next_state] = np.zeros(env.action_space.n)

        # Actualización de Q-learning
        best_next_q = np.max(q_table[next_state])  # Máximo valor Q del siguiente estado
        q_update_target = reward + params["discount_factor"] * best_next_q * (1 - terminated)
        q_table[state][action] += params["learning_rate"] * (q_update_target - q_table[state][action])

        # Actualizar estado y recompensa
        state = next_state
        total_reward += reward

    # Registrar recompensa
    rewards_all_episodes.append(total_reward)

    # Decaimiento exponencial de epsilon
    params["epsilon"] = params["min_epsilon"] + (params["max_epsilon"] - params["min_epsilon"]) * np.exp(-params["epsilon_decay_rate"] * episode)

    # Reportar progreso
    if (episode + 1) % params["report_interval"] == 0:
        avg_reward = np.mean(rewards_all_episodes[-params["report_interval"]:])
        print(f"Episodio {episode + 1:>6}/{params['num_episodes']} | "
              f"Recompensa promedio últimos {params['report_interval']}: {avg_reward:.3f} | "
              f"Epsilon: {params['epsilon']:.3f}")

env.close()
end_time = time.time()
print(f"Entrenamiento finalizado en {end_time - start_time:.2f} segundos.")

# --- Resultados ---

# Gráfico de recompensa promedio con media móvil
if len(rewards_all_episodes) >= params["moving_avg_window"]:
    rewards_moving_avg = np.convolve(
        rewards_all_episodes, np.ones(params["moving_avg_window"]) / params["moving_avg_window"], mode='valid'
    )
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(params["moving_avg_window"] - 1, params["num_episodes"]), rewards_moving_avg)
    plt.title(f"Recompensa Promedio en {params['env_name']} (Media Móvil {params['moving_avg_window']} episodios)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Promedio")
    plt.grid(True)
    plt.show()

# Mostrar algunos valores Q
sampled_states = list(q_table.keys())[:10]
print("\n--- Valores Q (primeros estados) ---")
for state in sampled_states:
    print(f"{state}: {q_table[state]}")

# Mostrar política aprendida para algunos estados relevantes
print("\n--- Tablas de Política Aprendida ---\n")
mano_jugador = list(range(12, 22))  # 12 a 21
mano_dealer = list(range(1, 11))    # 1 a 10

# Política cuando el As no es usable
print("Con As usable: False")
print("D / J", end=" ")  # Dealer ↓ / Jugador →
for mano_j in mano_jugador:
    print(f"{mano_j:>4}", end="")
print()

for dealer in mano_dealer:
    print(f"{dealer:>2} |", end=" ")
    for mano_j in mano_jugador:
        state = (mano_j, dealer, False)
        action = np.argmax(q_table[state]) if state in q_table else -1
        print(f"{action:>4}", end="")
    print()

# Política cuando el As es usable
print("\nCon As usable: True")
print("D / J", end=" ")  # Dealer ↓ / Jugador →
for mano_j in mano_jugador:
    print(f"{mano_j:>4}", end="")
print()

for dealer in mano_dealer:
    print(f"{dealer:>2} |", end=" ")
    for mano_j in mano_jugador:
        state = (mano_j, dealer, True)
        action = np.argmax(q_table[state]) if state in q_table else -1
        print(f"{action:>4}", end="")
    print()

print("\n--- Fin del análisis ---")