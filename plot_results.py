"""
=============================================================================
  plot_results.py  ·  Gráficas de análisis de entrenamiento (post-training)
  ---------------------------------------------------------------------------
  Genera el equivalente, para el modelo PPO del dron-péndulo, de las 4
  gráficas que MATLAB muestra automáticamente al entrenar una red neuronal
  con `train()` (nftool):

      MATLAB (supervisado)                 →   Aquí (PPO / RL)
      ------------------------------------------------------------------
      Performance (MSE train/val/test)     →   Recompensa media vs pasos
                                                (entrenamiento y evaluación)
      Training State (gradiente, mu,       →   Pérdidas de PPO (policy loss,
        validation checks)                     value loss, entropy, clip
                                                fraction, KL) vs pasos
      Error Histogram                      →   Histograma del error final
                                                (distancia carga→setpoint)
      Regression (Target vs Output)        →   Setpoint objetivo vs posición
                                                final alcanzada (x, z)

  Este script NO entrena nada. Solo:
    1. Lee los logs de TensorBoard que train.py ya generó en ./logs
    2. Lee evaluations.npz que el EvalCallback ya generó en ./logs
    3. Corre unos pocos episodios de INFERENCIA (no entrenamiento) con el
       modelo final para poder graficar el histograma de error y la
       "regresión" setpoint vs posición alcanzada.

  Requisitos: los mismos que train.py, más `tensorboard` (ya viene instalado
  si train.py corrió con tensorboard_log configurado).

  Uso:
      python plot_results.py
=============================================================================
"""

import os
import sys
import glob
import math
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from drone_env import DronePendulumEnv

SAVE_DIR         = os.path.join(SCRIPT_DIR, "models")
LOG_DIR          = os.path.join(SCRIPT_DIR, "logs")
MODEL_NAME       = "ppo_drone_pendulum"
N_EVAL_EPISODES  = 30       # episodios de inferencia para histograma/regresión
MAX_STEPS_EP     = 1500


# ─────────────────────────────────────────────────────────────────────────
#  1. Leer escalares de TensorBoard (curvas de entrenamiento)
# ─────────────────────────────────────────────────────────────────────────
def find_latest_tb_run(log_dir, model_name):
    candidates = sorted(glob.glob(os.path.join(log_dir, f"{model_name}_*")))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontró ningún run de TensorBoard en '{log_dir}'.\n"
            f"¿Corriste train.py primero?"
        )
    return candidates[-1]


def load_tb_scalars(tb_dir, tags):
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        sys.exit(
            "Falta el paquete 'tensorboard'. Instálalo con:\n"
            "    pip install tensorboard"
        )

    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])

    data = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            data[tag] = (
                np.array([e.step for e in events]),
                np.array([e.value for e in events]),
            )
        else:
            print(f"  (aviso: tag '{tag}' no encontrado en TensorBoard)")
            data[tag] = (np.array([]), np.array([]))
    return data


# ─────────────────────────────────────────────────────────────────────────
#  2. Leer evaluations.npz generado por el EvalCallback
# ─────────────────────────────────────────────────────────────────────────
def load_eval_npz(log_dir):
    path = os.path.join(log_dir, "evaluations.npz")
    if not os.path.exists(path):
        print(f"  (aviso: no se encontró {path})")
        return None
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"].mean(axis=1)   # media de las n_eval_episodes
    return timesteps, results


# ─────────────────────────────────────────────────────────────────────────
#  3. Correr episodios de INFERENCIA con el modelo ya entrenado
# ─────────────────────────────────────────────────────────────────────────
def run_inference_episodes(n_episodes=N_EVAL_EPISODES):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

    model_path = os.path.join(SAVE_DIR, "best_model")
    if not os.path.exists(model_path + ".zip"):
        model_path = os.path.join(SAVE_DIR, MODEL_NAME + "_final")
    if not os.path.exists(model_path + ".zip"):
        sys.exit(
            "No se encontró ningún modelo entrenado en ./models.\n"
            "Corre train.py primero."
        )

    vecnorm_path = os.path.join(SAVE_DIR, "vecnorm_stats.pkl")

    def _make():
        return DronePendulumEnv(render_mode=None)

    vec_env = DummyVecEnv([_make])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("  (aviso: no se encontró vecnorm_stats.pkl, sin normalizar obs)")

    model = PPO.load(model_path, device="auto")
    env = DronePendulumEnv(render_mode=None)

    x_sp, z_sp, x_final, z_final, dist_final, ep_reward = ([] for _ in range(6))

    print(f"  Corriendo {n_episodes} episodios de inferencia (sin render)…")
    for ep in range(n_episodes):
        obs, _ = env.reset()
        xsp_i, zsp_i = env.x_sp, env.z_sp
        total_r = 0.0
        done, steps = False, 0

        while not done and steps < MAX_STEPS_EP:
            obs_norm = vec_env.normalize_obs(np.array([obs]))
            action, _ = model.predict(obs_norm, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action[0])
            total_r += reward
            done = terminated or truncated
            steps += 1

        x, xd, z, zd, th, thd, ph, phd = env.state
        bob_x = x + math.sin(th)      # posición de la carga (l = 1.0 m)
        bob_z = z - math.cos(th)

        x_sp.append(xsp_i); z_sp.append(zsp_i)
        x_final.append(bob_x); z_final.append(bob_z)
        dist_final.append(math.hypot(bob_x - xsp_i, bob_z - zsp_i))
        ep_reward.append(total_r)

    env.close()
    return {
        "x_sp": np.array(x_sp), "z_sp": np.array(z_sp),
        "x_final": np.array(x_final), "z_final": np.array(z_final),
        "dist_final": np.array(dist_final), "ep_reward": np.array(ep_reward),
    }


# ─────────────────────────────────────────────────────────────────────────
#  4a. Figura 1 — "Performance" (curva de aprendizaje)
# ─────────────────────────────────────────────────────────────────────────
def plot_performance(tb_data, eval_data):
    steps_tr, rew_tr = tb_data["rollout/ep_rew_mean"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.canvas.manager.set_window_title("Performance")

    if len(steps_tr):
        ax.plot(steps_tr, rew_tr, color="tab:blue", label="Training (rollout)")

    best_txt = ""
    if eval_data is not None:
        steps_ev, rew_ev = eval_data
        ax.plot(steps_ev, rew_ev, color="tab:green", label="Validation (eval)")
        best_i = int(np.argmax(rew_ev))
        ax.axvline(steps_ev[best_i], color="gray", linestyle=":", linewidth=1)
        ax.plot(steps_ev[best_i], rew_ev[best_i], "o", color="tab:green",
                 markersize=9, markerfacecolor="none", markeredgewidth=2)
        best_txt = f"Best Eval Reward is {rew_ev[best_i]:.1f} at step {int(steps_ev[best_i]):,}"

    ax.set_title(best_txt or "Training Performance")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Recompensa media por episodio")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()


# ─────────────────────────────────────────────────────────────────────────
#  4b. Figura 2 — "Training State" (pérdidas / métricas de PPO)
# ─────────────────────────────────────────────────────────────────────────
def plot_training_state(tb_data):
    panels = [
        ("train/value_loss",           "Value Loss"),
        ("train/policy_gradient_loss", "Policy Gradient Loss"),
        ("train/entropy_loss",         "Entropy Loss"),
        ("train/clip_fraction",        "Clip Fraction"),
        ("train/approx_kl",            "Approx. KL"),
        ("train/explained_variance",   "Explained Variance"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 11), sharex=True)
    fig.canvas.manager.set_window_title("Training State")

    for ax, (tag, label) in zip(axes, panels):
        steps, vals = tb_data[tag]
        if len(steps):
            ax.plot(steps, vals, color="tab:blue", linewidth=1.3)
            last_val = vals[-1]
            ax.set_title(f"{label} = {last_val:.4g} (último punto)", fontsize=10)
        else:
            ax.set_title(f"{label} (sin datos)", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylabel(label, fontsize=8)

    axes[-1].set_xlabel("Timesteps")
    fig.tight_layout()


# ─────────────────────────────────────────────────────────────────────────
#  4c. Figura 3 — "Error Histogram"
# ─────────────────────────────────────────────────────────────────────────
def plot_error_histogram(infer):
    dist = infer["dist_final"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.canvas.manager.set_window_title("Error Histogram")

    ax.hist(dist, bins=20, color="tab:blue", edgecolor="white",
             label=f"Distancia final al setpoint (N={len(dist)})")
    ax.axvline(0.0, color="orange", linewidth=1.5, label="Zero Error")
    ax.axvline(np.mean(dist), color="tab:red", linestyle="--",
                label=f"Media = {np.mean(dist):.3f} m")

    ax.set_title("Error Histogram — distancia carga→setpoint al final del episodio")
    ax.set_xlabel("Error = |posición final − setpoint|  (m)")
    ax.set_ylabel("Nº de episodios")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()


# ─────────────────────────────────────────────────────────────────────────
#  4d. Figura 4 — "Regression" (Setpoint vs posición alcanzada)
# ─────────────────────────────────────────────────────────────────────────
def _fit_and_plot(ax, target, output, title, color):
    ax.scatter(target, output, facecolors="none", edgecolors=color,
               label="Data", zorder=3)
    lims = [min(target.min(), output.min()) - 0.5,
            max(target.max(), output.max()) + 0.5]
    ax.plot(lims, lims, "k--", linewidth=1, label="Y = T")

    if len(target) >= 2 and np.std(target) > 1e-9:
        m, b = np.polyfit(target, output, 1)
        r = np.corrcoef(target, output)[0, 1]
        xs = np.array(lims)
        ax.plot(xs, m * xs + b, color=color, linewidth=2,
                 label=f"Fit: Output = {m:.2f}·Target + {b:.2f}")
        title = f"{title}: R={r:.5f}"

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Target (setpoint)")
    ax.set_ylabel("Output (posición final)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)


def plot_regression(infer):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.canvas.manager.set_window_title("Regression")

    _fit_and_plot(axes[0], infer["x_sp"], infer["x_final"], "Eje X", "tab:blue")
    _fit_and_plot(axes[1], infer["z_sp"], infer["z_final"], "Eje Z", "tab:green")

    all_target = np.concatenate([infer["x_sp"], infer["z_sp"]])
    all_output = np.concatenate([infer["x_final"], infer["z_final"]])
    _fit_and_plot(axes[2], all_target, all_output, "Todos (X+Z)", "tab:gray")

    fig.tight_layout()


# ─────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  Drone-Pendulum DRL — Gráficas de análisis de entrenamiento")
    print("=" * 65)

    print("\n  [1/3] Leyendo logs de TensorBoard…")
    tb_dir = find_latest_tb_run(LOG_DIR, MODEL_NAME)
    print(f"        Run encontrado: {tb_dir}")
    tags = [
        "rollout/ep_rew_mean",
        "train/value_loss",
        "train/policy_gradient_loss",
        "train/entropy_loss",
        "train/clip_fraction",
        "train/approx_kl",
        "train/explained_variance",
    ]
    tb_data = load_tb_scalars(tb_dir, tags)

    print("\n  [2/3] Leyendo evaluations.npz (EvalCallback)…")
    eval_data = load_eval_npz(LOG_DIR)

    print("\n  [3/3] Ejecutando episodios de inferencia para histograma/regresión…")
    infer = run_inference_episodes(N_EVAL_EPISODES)

    print("\n  Generando figuras…")
    plot_performance(tb_data, eval_data)
    plot_training_state(tb_data)
    plot_error_histogram(infer)
    plot_regression(infer)

    print("  Listo. Cierra las ventanas para terminar.\n")
    plt.show()