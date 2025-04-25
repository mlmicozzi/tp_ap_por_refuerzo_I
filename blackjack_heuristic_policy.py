import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

# --- Parámetros ---
env_name = "Blackjack-v1"

# Evaluación de la política heurística
num_episodes = 200000   # Número de episodios para entrenar

# Para reporte/gráficas
report_interval = num_episodes // 20  # Cada cuántos episodios mostrar progreso
moving_avg_window = 5000  # Ventana para la media móvil en la gráfica

# --- Inicialización ---
env = gym.make(env_name)

# --- Política Heurística ---
def heuristic_policy(state):
    """ Política heurística simple para Blackjack """
    player_hand, dealer_card, _ = state
    
    # Si el jugador tiene un total de 17 o más, se queda
    if player_hand >= 17:
        return 0  # 0 -> Quedarse (Stick)
    
    # Si el jugador tiene un total menor que 12, siempre pide
    if player_hand <= 11:
        return 1  # 1 -> Pedir (Hit)
    
    # Si el jugador tiene entre 12 y 16, decide en base a la carta del dealer
    if 12 <= player_hand <= 16:
        if dealer_card <= 6:
            return 0  # Quedarse si el dealer tiene carta baja
        else:
            return 1  # Pedir si el dealer tiene carta alta

    return 0  # Por defecto quedarse


# --- Evaluación de la política heurística ---
def evaluate_heuristic_policy(env, num_episodes):
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = heuristic_policy(state)  # Obtener la acción según la política heurística
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        total_rewards.append(total_reward)
        
        # Mostrar el avance cada ciertos episodios
        if (episode + 1) % report_interval == 0:
            avg_reward = np.mean(total_rewards[-report_interval:])
            print(f"Evaluación Episodio {episode+1}/{num_episodes} | "
                  f"Recompensa Promedio: {avg_reward:.3f}")

    avg_reward = np.mean(total_rewards)
    print(f"Recompensa promedio con la política heurística: {avg_reward:.3f}")
    return total_rewards  # Ahora devolvemos todas las recompensas de los episodios

# Mostrar política según la heurística
print("\n--- Tablas de Política Heurística ---\n")
cols = list(range(12, 22))
print("D / J ", end="")
for col in cols:
    print(f"{col:>3}", end="")
print()

for dealer in range(1, 11):
    print(f"{dealer:>2} | ", end="")
    for player_sum in cols:
        action = heuristic_policy((player_sum, dealer, False)) 
        print(f"{action:>3}", end="")
    print()

print("\n------------------")

# --- Reporte del rendimiento ---
start_time = time.time()

# Evaluar la política heurística y guardar las recompensas
rewards_all_episodes = evaluate_heuristic_policy(env, num_episodes=num_episodes)

env.close()
end_time = time.time()
print(f"Evaluación finalizada en {end_time - start_time:.2f} segundos.")

# --- Resultados ---

# Gráfico de recompensa con media móvil
if len(rewards_all_episodes) >= moving_avg_window:
    rewards_moving_avg = np.convolve(
        rewards_all_episodes, np.ones(moving_avg_window) / moving_avg_window, mode='valid'
    )
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(moving_avg_window - 1, num_episodes), rewards_moving_avg)
    plt.title(f"Recompensa Promedio con Política Heurística en {env_name} (Media Móvil {moving_avg_window} episodios)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa Promedio")
    plt.grid(True)
    plt.show()

print("\n--- Fin del análisis ---")