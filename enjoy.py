"""
=============================================================================
  enjoy.py  ·  Script de Visualización / Inferencia
  ---------------------------------------------------------------------------
  • Si existe un modelo entrenado → lo carga y lo ejecuta gráficamente.
  • Si NO existe modelo           → corre un modo DEMO básico para que
    puedas ver la física del dron y el renderizado en tiempo real.
  • La ventana NO se cierra automáticamente al terminar el episodio;
    espera a que presiones ESC/Q o cierres la ventana manualmente.
=============================================================================
"""

import os
import sys
import time
import numpy as np

# ── Ajusta la ruta para encontrar drone_env.py en el mismo directorio ────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from drone_env import DronePendulumEnv, U_F_HOVER, U_F_MAX, U_TAU_MAX

SAVE_DIR   = os.path.join(SCRIPT_DIR, "models")
MODEL_PATH = os.path.join(SAVE_DIR, "best_model")
VECNORM    = os.path.join(SAVE_DIR, "vecnorm_stats.pkl")


# ─────────────────────────────────────────────────────────────────────────────
#  MODO DEMO  (Sin modelo entrenado)
# ─────────────────────────────────────────────────────────────────────────────
def hover_action(step: int) -> np.ndarray:
    """
    IA/Control: Simula una acción de control básica (Vuelo estático + balanceo).
      a[0] → Empuje (thrust) fijo igual al peso para flotar.
      a[1] → Torque que varía suavemente en el tiempo (onda senoidal).
    """
    # Empuje normalizado
    a0 = np.clip(2.0 * U_F_HOVER / U_F_MAX - 1.0, -1.0, 1.0)
    # Torque suave y oscilante
    a1 = 0.15 * np.sin(step * 0.03)
    return np.array([a0, a1], dtype=np.float32)


def run_demo(n_episodes: int = 3, steps_per_ep: int = 1500):
    """Ciclo de Vida: Ejecuta episodios de prueba usando una política de control fija."""
    print("\n  ── MODO DEMO (no hay modelo entrenado) ──")
    print("  Presiona Ctrl+C en consola o cierra la ventana gráfica para salir.\n")

    env = DronePendulumEnv(render_mode="human")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_r = 0.0
        print(f"  Episodio {ep}/{n_episodes}  |  Objetivo → x={env.x_sp:.2f} m  z={env.z_sp:.2f} m")

        for step in range(steps_per_ep):
            action = hover_action(step)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += reward

            if terminated or truncated:
                break

        print(f"    Pasos: {env.step_num:<5d}  Recompensa total: {total_r:.1f}")

    print("\n  Demo finalizado. Cerrando ventana…")
    _wait_for_close(env)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
#  MODO INFERENCIA  (Con modelo IA entrenado)
# ─────────────────────────────────────────────────────────────────────────────
def run_trained(n_episodes: int = 5, steps_per_ep: int = 1500):
    """IA/Aprendizaje: Carga la red neuronal entrenada (PPO) y la ejecuta."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    print(f"\n  Cargando modelo de inteligencia artificial desde:  {MODEL_PATH}.zip")

    # Entorno principal para ver la física
    env = DronePendulumEnv(render_mode="human")

    # Entorno auxiliar fantasma, usado puramente para escalar/normalizar los sensores
    def _make():
        return DronePendulumEnv(render_mode=None)

    vec_env = DummyVecEnv([_make])

    if os.path.exists(VECNORM):
        vec_env = VecNormalize.load(VECNORM, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"  Estadísticas de normalización cargadas desde: {VECNORM}")
    else:
        print("  (No se encontró vecnorm_stats.pkl — se usarán sensores sin normalizar)")

    model = PPO.load(MODEL_PATH, device="auto")
    print("  Modelo cargado correctamente.\n")

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        total_r = 0.0
        print(f"  Episodio {ep}/{n_episodes}  |  Objetivo → x={env.x_sp:.2f} m  z={env.z_sp:.2f} m")

        for step in range(steps_per_ep):
            # Normalizar los datos de los sensores antes de pasarlos al cerebro (modelo)
            obs_norm = vec_env.normalize_obs(np.array([obs]))
            
            # La IA predice la mejor acción
            action, _ = model.predict(obs_norm, deterministic=True)
            
            # Aplicar los comandos a los motores del dron físico
            obs, reward, terminated, truncated, _ = env.step(action[0])
            total_r += reward

            if terminated or truncated:
                # Si el dron choca, pausar 1.5 segundos para que el usuario alcance a ver el choque
                import pygame
                pygame.time.wait(1500)
                break

        print(f"    Pasos: {env.step_num:<5d}  Recompensa total: {total_r:.1f}")

    print("\n  Inferencia finalizada. Cerrando ventana…")
    _wait_for_close(env)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIÓN AUXILIAR
# ─────────────────────────────────────────────────────────────────────────────
def _wait_for_close(env: DronePendulumEnv, timeout: float = 30.0):
    """
    Renderizado: Congela la ventana de la simulación al finalizar el último episodio
    para que el usuario tenga tiempo de ver el estado final sin que desaparezca.
    """
    import pygame
    t0 = time.time()
    print("  (Cierra la ventana de Pygame o espera 30 s para salir)")
    while time.time() - t0 < timeout:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return
            if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q):
                return
        pygame.time.wait(50)


# ─────────────────────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  Dron-Péndulo DRL  ·  Script de Visualización")
    print("=" * 62)

    model_exists = os.path.exists(MODEL_PATH + ".zip")

    if not model_exists:
        print(f"\n  ⚠  No se encontró una Inteligencia Artificial en '{MODEL_PATH}.zip'")
        print("  → Para entrenarla, ejecuta primero:  python train.py")
        print("  → Mientras tanto, mostrando animación de demostración (MODO DEMO)…")
        run_demo(n_episodes=3, steps_per_ep=1500)
    else:
        run_trained(n_episodes=5, steps_per_ep=1500)

    print("\n  Fin del programa.")
