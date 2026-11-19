"""
=============================================================================
  train.py  ·  Script de Entrenamiento PPO
  Entrena un agente PPO de Stable-Baselines3 en DronePendulumEnv.
=============================================================================
"""

import os
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from drone_env import DronePendulumEnv

# ─────────────────────────────────────────────────────────────────────────────
#  Rutas de guardado
# ─────────────────────────────────────────────────────────────────────────────
SAVE_DIR   = "./models"
LOG_DIR    = "./logs"
MODEL_NAME = "ppo_drone_pendulum"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Callback personalizado: imprime estadísticas cada N episodios
# ─────────────────────────────────────────────────────────────────────────────
class EpisodeLogCallback(BaseCallback):
    """Métricas: Imprime la recompensa media y longitud del episodio cada K episodios."""

    def __init__(self, log_freq_ep: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq_ep   = log_freq_ep
        self.ep_count      = 0
        self.ep_rewards:   list = []
        self.ep_lengths:   list = []
        self._ep_reward_buf = 0.0
        self._ep_len_buf    = 0

    def _on_step(self) -> bool:
        # Acumular recompensa del episodio
        reward = self.locals["rewards"][0]
        done   = self.locals["dones"][0]
        self._ep_reward_buf += reward
        self._ep_len_buf    += 1

        if done:
            self.ep_count += 1
            self.ep_rewards.append(self._ep_reward_buf)
            self.ep_lengths.append(self._ep_len_buf)
            self._ep_reward_buf = 0.0
            self._ep_len_buf    = 0

            if self.ep_count % self.log_freq_ep == 0:
                mean_r = np.mean(self.ep_rewards[-self.log_freq_ep:])
                mean_l = np.mean(self.ep_lengths[-self.log_freq_ep:])
                print(
                    f"  [Ep {self.ep_count:>5d}]  "
                    f"MeanRew={mean_r:>9.2f}  "
                    f"MeanLen={mean_l:>6.1f}  "
                    f"TotalSteps={self.num_timesteps:>8d}"
                )
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  Punto de entrada principal para el entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TOTAL_TIMESTEPS = 1_500_000
    N_ENVS          = 4           # Entornos paralelos para recolectar datos rápido
    EVAL_FREQ       = 10_000      # Evaluar el modelo cada N pasos
    SAVE_FREQ       = 20_000      # Guardar un checkpoint cada N pasos

    print("=" * 65)
    print("  Dron-Péndulo DRL — Entrenamiento PPO con Stable-Baselines3")
    print("=" * 65)

    # ── Crear entornos de entrenamiento vectorizados ───────────────────────────
    print(f"\n  Creando {N_ENVS} entornos de entrenamiento en paralelo …")
    train_env = make_vec_env(
        DronePendulumEnv,
        n_envs=N_ENVS,
        seed=42,
        env_kwargs={"render_mode": None},   # Sin renderizado visual durante el entrenamiento
    )

    # Vectorización: Envuelve el entorno con VecNormalize para normalizar observaciones y recompensas.
    # Esto ayuda muchísimo a la estabilidad de PPO en tareas de control continuo.
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # ── Crear un entorno separado para evaluación ──────────────────────────────
    eval_env_raw = make_vec_env(
        DronePendulumEnv,
        n_envs=1,
        seed=99,
        env_kwargs={"render_mode": None},
    )
    eval_env = VecNormalize(
        eval_env_raw,
        training=False,      # ¡Importante! No actualizar las estadísticas medias durante la evaluación
        norm_obs=True,
        norm_reward=False,   # Evaluar basándose en la recompensa pura, sin normalizar
    )

    # ── Callbacks (Rutinas automáticas) ────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=SAVE_DIR,
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1),
        save_path=SAVE_DIR,
        name_prefix=MODEL_NAME,
        save_vecnormalize=True,
    )
    log_cb = EpisodeLogCallback(log_freq_ep=20)

    # ── Hiperparámetros de PPO ─────────────────────────────────────────────────
    #   Ajustados para una tarea moderadamente compleja de control continuo.
    model = PPO(
        policy         = "MlpPolicy",
        env            = train_env,
        learning_rate  = 3e-4,
        n_steps        = 1024,            # Pasos recopilados por entorno antes de actualizar la red
        batch_size     = 256,             # Tamaño del mini-lote para el gradiente
        n_epochs       = 10,              # Épocas de gradiente por actualización
        gamma          = 0.99,            # Factor de descuento del futuro
        gae_lambda     = 0.95,            # GAE lambda (estimación de ventajas)
        clip_range     = 0.2,             # Parámetro de recorte PPO
        clip_range_vf  = None,            # Sin recorte en la función de valor
        ent_coef       = 0.015,           # Bono de entropía (fomenta la exploración aleatoria)
        vf_coef        = 0.5,             # Coeficiente de pérdida de la función de valor
        max_grad_norm  = 0.5,             # Recorte de gradientes para evitar explosiones
        normalize_advantage = True,
        policy_kwargs  = dict(
            net_arch   = [dict(pi=[256, 256], vf=[256, 256])],  # Red Neuronal: 2 capas ocultas de 256
        ),
        tensorboard_log = LOG_DIR,
        verbose         = 1,
        seed            = 42,
        device          = "auto",         # Usará la GPU (CUDA) si la tienes
    )

    # ── Entrenar ───────────────────────────────────────────────────────────────
    print(f"\n  Iniciando entrenamiento por {TOTAL_TIMESTEPS:,} pasos de tiempo …")
    print(f"  Red neuronal : MlpPolicy  [256, 256] × 2 (pi + vf)")
    print(f"  Sensores     : {train_env.observation_space.shape}")
    print(f"  Motores      : {train_env.action_space.shape}")
    print()

    t0 = time.time()
    model.learn(
        total_timesteps       = TOTAL_TIMESTEPS,
        callback              = [eval_cb, ckpt_cb, log_cb],
        log_interval          = 1,
        tb_log_name           = MODEL_NAME,
        reset_num_timesteps   = True,
        progress_bar          = True,
    )
    elapsed = time.time() - t0

    # ── Guardar modelo final y estadísticas de normalización ───────────────────
    final_model_path = os.path.join(SAVE_DIR, MODEL_NAME + "_final")
    model.save(final_model_path)
    train_env.save(os.path.join(SAVE_DIR, "vecnorm_stats.pkl"))

    print("\n" + "=" * 65)
    print(f"  Entrenamiento completado en {elapsed:.1f} s  ({elapsed/60:.1f} min)")
    print(f"  Modelo final guardado  →  {final_model_path}.zip")
    print(f"  Mejor modelo guardado  →  {SAVE_DIR}/best_model.zip")
    print("=" * 65)

    train_env.close()
    eval_env.close()