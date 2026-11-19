"""
=============================================================================
  DronePendulumEnv  ·  drone_env.py
  Entorno personalizado de Gymnasium: Cuadricóptero 2D con carga suspendida
  ---------------------------------------------------------------------------
  Física     : Formulación de Euler-Lagrange del sistema acoplado dron-péndulo
  Renderizado: Animación en tiempo real con Pygame-CE (render_mode="human")
  Entrenamiento: Stable-Baselines3 PPO (MlpPolicy)
=============================================================================
"""

import sys
import math
import warnings
import gymnasium as gym
from gymnasium import spaces
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES FÍSICAS
# ─────────────────────────────────────────────────────────────────────────────
M   = 1.5          # Masa del cuerpo del dron               [kg]
m   = 0.5          # Masa de la carga suspendida (bola)     [kg]
Mt  = M + m        # Masa total del sistema                 [kg]
l   = 1.0          # Longitud del cable del péndulo         [m]
g   = 9.81         # Aceleración gravitacional              [m/s²]
I   = 0.01         # Momento de inercia del dron (balanceo) [kg·m²]

# Coeficientes de resistencia aerodinámica y amortiguamiento
C_X     = 0.15     # Resistencia traslacional horizontal    [N·s/m]
C_Z     = 0.15     # Resistencia traslacional vertical      [N·s/m]
C_PHI   = 0.25     # Amortiguamiento del balanceo (roll)    [N·m·s/rad]
C_THETA = 0.02     # Amortiguamiento del péndulo            [N·m·s/rad]

# Empuje de sustentación (hover): u_f·cos(0) = Mt·g
U_F_HOVER = Mt * g   # ≈ 19.62 N

# Límites físicos de las acciones de control
U_F_MAX   = 50.0   # Empuje (thrust) máximo [N]
U_TAU_MAX = 0.4    # Torque máximo          [N·m]

# Paso de tiempo de la simulación (Bucle de control a 50 Hz)
DT = 0.02          # [s]

# Límites del espacio de trabajo y condiciones de terminación (choque)
X_BOUND = 12.0
Z_MIN   = 0.0
Z_MAX   = 12.0
PHI_MAX = math.pi / 3.0   # ±60 ° de inclinación → se considera un choque

MAX_STEPS = 1500

# Parámetros del modelo de ráfagas de viento
WIND_P_START = 0.05       # Probabilidad de inicio de ráfaga
WIND_F_MAX   = 5.0        # Fuerza máxima del viento [N]
WIND_DUR_MIN = 0.5        # Duración mínima [s]
WIND_DUR_MAX = 2.0        # Duración máxima [s]

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTES DE VISUALIZACIÓN DE PYGAME
# ─────────────────────────────────────────────────────────────────────────────
SCREEN_W  = 1100
SCREEN_H  = 720
FPS       = 50

# Rango de coordenadas del mundo mapeadas a la pantalla
WORLD_X   = (-X_BOUND, X_BOUND)          # Metros → píxeles horizontales
WORLD_Z   = (Z_MIN - 0.5, Z_MAX + 0.5)   # Metros → píxeles verticales

# Paleta de colores (RGB)
C_BG         = (13,  17,  26)
C_GRID       = (22,  32,  55)
C_GROUND     = (35,  75,  45)
C_DRONE      = (30, 160,  70)
C_DRONE_EDGE = (80, 220, 110)
C_ROTOR      = (50, 200, 100)
C_CABLE      = (160, 195, 230)
C_BOB        = (80,  195, 255)
C_BOB_SHINE  = (200, 240, 255)
C_SETPOINT   = (255, 215,  50)
C_THRUST_ARR = (255, 110,  20)
C_WIND_ARR   = (255, 175,  40)
C_HUD_BG     = (8,   12,  25)
C_HUD_BORDER = (35,  50,  80)
C_HUD_TITLE  = (100, 210, 120)
C_HUD_TXT    = (185, 200, 220)
C_HUD_SP     = (110, 165, 255)
C_HUD_ERR    = (255, 160,  80)
C_BAR_BG     = (25,  35,  55)
C_BAR_FG     = (60,  180, 110)
C_COM        = (255,  80,  80)


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────
def _rotate_points(pts, cx, cy, angle_rad):
    """
    Geometría: Rota una lista de puntos (x,y) alrededor de un centro (cx, cy).
    Se usa para dibujar el dron inclinado en la pantalla.
    """
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    result = []
    for (px, py) in pts:
        dx, dy = px - cx, py - cy
        result.append((
            cx + dx * cos_a - dy * sin_a,
            cy + dx * sin_a + dy * cos_a,
        ))
    return result


class DronePendulumEnv(gym.Env):
    """
    Entorno personalizado de Gymnasium para un cuadricóptero 2D que lleva una
    carga de péndulo.

    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  ESTADO  (8 componentes, interno)                                        ║
    ║  s = [ x, ẋ, z, ż, θ, θ̇, φ, φ̇ ]                                          ║
    ║       x   : Posición horizontal del dron     [m]                         ║
    ║       ẋ   : Velocidad horizontal del dron    [m/s]                       ║
    ║       z   : Altitud del dron                 [m]                         ║
    ║       ż   : Velocidad vertical del dron      [m/s]                       ║
    ║       θ   : Ángulo del péndulo (desde −ẑ, CCW+)  [rad]                   ║
    ║       θ̇   : Velocidad angular del péndulo    [rad/s]                     ║
    ║       φ   : Ángulo de balanceo del dron      [rad]                       ║
    ║       φ̇   : Velocidad de balanceo del dron   [rad/s]                     ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  OBSERVACIÓN  (10-D devuelta al agente)                                  ║
    ║  obs = [ x, ẋ, z, ż, θ, θ̇, φ, φ̇, x_sp, z_sp ]                            ║
    ╠══════════════════════════════════════════════════════════════════════════╣
    ║  ACCIÓN  (2-D continua ∈ [-1,1])                                         ║
    ║  a[0] → u_f  = (a[0]+1)/2 · U_F_MAX   ∈ [0, 40] N                        ║
    ║  a[1] → u_τ  = a[1] · U_TAU_MAX       ∈ [-0.2, 0.2] N·m                  ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """

    metadata = {"render_modes": ["human"], "render_fps": FPS}

    # ── Constructor ─────────────────────────────────────────────────────────
    def __init__(self, render_mode: str = None):
        """
        Inicialización: Define los espacios de acción y observación de OpenAI Gym.
        Prepara las variables internas.
        """
        super().__init__()
        self.render_mode = render_mode

        # Espacio de Observación: [x, ẋ, z, ż, θ, θ̇, φ, φ̇, x_sp, z_sp]
        obs_lo = np.array([-X_BOUND, -20, Z_MIN, -20, -math.pi, -20, -PHI_MAX, -20, -X_BOUND, Z_MIN], dtype=np.float32)
        obs_hi = np.array([ X_BOUND,  20, Z_MAX,  20,  math.pi,  20,  PHI_MAX,  20,  X_BOUND, Z_MAX], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_lo, high=obs_hi, dtype=np.float32)

        # Espacio de Acción: [a_thrust, a_torque] ∈ [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Estado interno
        self.state:       np.ndarray = None
        self.x_sp:        float      = 0.0
        self.z_sp:        float      = 5.0
        self.step_num:    int        = 0
        self.prev_action: np.ndarray = np.zeros(2, dtype=np.float32)
        self._last_uf:    float      = U_F_HOVER   # Para escalar la flecha visual de empuje

        # Estado del modelo de viento
        self._wind_force: float = 0.0
        self._wind_steps: int   = 0
        self._wind_drone: float = 0.0
        self._wind_load:  float = 0.0

        # Estado de Pygame (inicialización diferida en la primera llamada de render)
        self._pg_init:  bool = False
        self._screen          = None
        self._clock           = None
        self._font_sm         = None
        self._font_md         = None
        self._font_lg         = None

    # ══════════════════════════════════════════════════════════════════════
    #  RESETEO
    # ══════════════════════════════════════════════════════════════════════
    def reset(self, *, seed=None, options=None):
        """
        Ciclo de Vida: Reinicia la simulación para un nuevo episodio.
        Coloca el dron en una posición inicial aleatoria y genera un nuevo objetivo (setpoint).
        """
        super().reset(seed=seed)
        rng = self.np_random

        # Estado inicial semi-aleatorio
        self.state = np.array([
            rng.uniform(-1.0,  1.0),    # x
            rng.uniform(-0.1,  0.1),    # ẋ
            rng.uniform( 4.0,  6.0),    # z
            rng.uniform(-0.1,  0.1),    # ż
            rng.uniform(-0.08, 0.05),   # θ
            rng.uniform(-0.02, 0.02),   # θ̇
            rng.uniform(-0.05, 0.05),   # φ
            0.0,                        # φ̇
        ], dtype=np.float64)

        # Posición objetivo a alcanzar (setpoint)
        self.x_sp = 4
        self.z_sp = 6.5
        
        self.step_num    = 0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self._last_uf    = U_F_HOVER
        self._wind_force = self._wind_drone = self._wind_load = 0.0
        self._wind_steps = 0

        return self._get_obs(), {}

    # ══════════════════════════════════════════════════════════════════════
    #  PASO DE SIMULACIÓN (STEP)
    # ══════════════════════════════════════════════════════════════════════
    def step(self, action: np.ndarray):
        """
        Control RL: Recibe las acciones de los motores de la IA, avanza el motor 
        de física 1 paso de tiempo, y devuelve las observaciones y la recompensa.
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float64)

        # Desnormalizar acciones [-1, 1] hacia unidades físicas reales [Newtons]
        u_f   = (action[0] + 1.0) * 0.5 * U_F_MAX
        u_tau = action[1] * U_TAU_MAX
        self._last_uf = u_f

        # Ejecutar física
        self._update_wind()
        self.state = self._euler_step(self.state, u_f, u_tau)
        self.step_num += 1

        # Calcular desempeño (reward)
        reward, terminated = self._compute_reward(action, u_f, u_tau)
        truncated = (self.step_num >= MAX_STEPS)
        self.prev_action = action.astype(np.float32)

        if self.render_mode == "human":
            self.render()

        info = {
            "x": self.state[0], "z": self.state[2],
            "theta_deg": math.degrees(self.state[4]),
            "phi_deg":   math.degrees(self.state[6]),
            "wind_N":    self._wind_force,
        }

        return self._get_obs(), reward, terminated, truncated, info

    # ══════════════════════════════════════════════════════════════════════
    #  MOTOR DE FÍSICA
    # ══════════════════════════════════════════════════════════════════════
    def _dynamics(self, state: np.ndarray, u_f: float, u_tau: float) -> np.ndarray:
        """
        Cálculos de Física: Evalúa las aceleraciones del sistema usando el modelo
        de masa acoplada de Euler-Lagrange. Resuelve el sistema lineal M * a = F.
        """
        x, xd, z, zd, th, thd, ph, phd = state

        s_th = math.sin(th);  c_th = math.cos(th)
        s_ph = math.sin(ph);  c_ph = math.cos(ph)

        # Matriz de masa 3x3 (acoplamiento entre posición del dron y ángulo del péndulo)
        M_mat = np.array([
            [Mt,    0.0,  m * l * c_th],
            [0.0,   Mt,   m * l * s_th],
            [c_th,  s_th, l           ],
        ])

        # Fuerzas centrífugas generadas por la rotación del péndulo
        cent_x =  m * l * thd**2 * s_th
        cent_z = -m * l * thd**2 * c_th

        # Componentes de fuerza de los motores (inclinadas por el roll del dron)
        Fx_thrust = u_f * s_ph
        Fz_thrust = u_f * c_ph

        # Vector de fuerzas
        f_vec = np.array([
            Fx_thrust + self._wind_drone + self._wind_load - C_X * xd + cent_x,
            Fz_thrust - Mt * g           - C_Z * zd        + cent_z,
            -g * s_th + (self._wind_load * c_th) / (m + 1e-9)
                      - (C_THETA * thd)          / (m * l + 1e-9),
        ])

        # Resolver el sistema para encontrar las aceleraciones (x_dd, z_dd, th_dd)
        try:
            acc = np.linalg.solve(M_mat, f_vec)
        except np.linalg.LinAlgError:
            acc = np.linalg.lstsq(M_mat, f_vec, rcond=None)[0]

        x_dd, z_dd, th_dd = acc
        
        # El balanceo (roll) está desacoplado (aceleración angular pura)
        ph_dd = (u_tau - C_PHI * phd) / I

        return np.array([xd, x_dd, zd, z_dd, thd, th_dd, phd, ph_dd])

    def _euler_step(self, state: np.ndarray, u_f: float, u_tau: float) -> np.ndarray:
        """
        Cálculos de Física: Integración numérica de Euler explícito.
        Actualiza el estado sumándole la derivada multiplicada por el diferencial de tiempo.
        """
        ns = state + DT * self._dynamics(state, u_f, u_tau)
        
        # Mantener los ángulos normalizados entre [-π, π]
        ns[4] = math.atan2(math.sin(ns[4]), math.cos(ns[4]))
        ns[6] = math.atan2(math.sin(ns[6]), math.cos(ns[6]))
        
        # Detectar contacto con el suelo
        if ns[2] < Z_MIN:
            ns[2] = Z_MIN
            ns[3] = max(0.0, ns[3])
        return ns

    # ══════════════════════════════════════════════════════════════════════
    #  MODELO DE VIENTO
    # ══════════════════════════════════════════════════════════════════════
    def _update_wind(self):
        """
        Modelo Ambiental: Genera ráfagas de viento estocásticas (aleatorias).
        El viento empuja tanto al cuerpo del dron como a la carga.
        """
        if self._wind_steps <= 0:
            if self.np_random.random() < WIND_P_START:
                force = self.np_random.uniform(-WIND_F_MAX, WIND_F_MAX)
                dur   = self.np_random.uniform(WIND_DUR_MIN, WIND_DUR_MAX)
                self._wind_steps = max(1, int(dur / DT))
                self._wind_force = force
                self._wind_drone = force * 0.70
                self._wind_load  = force * 0.30
            else:
                self._wind_force = self._wind_drone = self._wind_load = 0.0
        else:
            self._wind_steps -= 1
            if self._wind_steps <= 0:
                self._wind_force = self._wind_drone = self._wind_load = 0.0

    # ══════════════════════════════════════════════════════════════════════
    #  SISTEMA DE RECOMPENSAS (REWARD)
    # ══════════════════════════════════════════════════════════════════════
    def _compute_reward(self, action: np.ndarray, u_f: float, u_tau: float):
        """
        IA/Aprendizaje: Calcula qué tan bien lo está haciendo el dron.
        Un puntaje mayor indica que la carga está más cerca del objetivo.
        """
        x, xd, z, zd, th, thd, ph, phd = self.state

        # 1. Fuera de Límites (Choque mortal)
        # Nota: Quitamos el crash por péndulo (THETA_MAX) porque el péndulo oscilando 
        # no debe destruir mágicamente el dron, solo desestabilizarlo físicamente.
        if abs(x) > X_BOUND or z < Z_MIN or z > Z_MAX or abs(ph) > PHI_MAX:
            return -100.0, True

        # Posición absoluta de la carga (péndulo) en el espacio
        bob_x = x + l * math.sin(th)
        bob_z = z - l * math.cos(th)

        # El error ahora se mide desde la carga hasta el setpoint
        dx, dz   = bob_x - self.x_sp, bob_z - self.z_sp
        dist     = math.hypot(dx, dz)

        # ── Función de Recompensa Positiva ──
        # Dar recompensas estrictamente positivas (de 0 a 1) por sobrevivir y acercarse
        # fomenta un comportamiento robusto y evita políticas suicidas.

        # 2. Bono de Distancia a la meta (Exponencial, de 0 a 5.0)
        r_dist = 5.0 * math.exp(-0.5 * dist)
        
        # 3. Bono de Estabilidad (0 a 1.0 cada uno)
        # Premia al dron por estar horizontal y al péndulo por estar quieto
        r_upright  = math.exp(-2.0 * abs(ph))
        r_pendulum = math.exp(-1.0 * abs(th))
        
        # 4. Bono de Vuelo Estático / Freno (0 a 1.0)
        # Fomenta que el dron se detenga y no se mueva a lo loco
        r_hover = math.exp(-0.2 * (xd**2 + zd**2))
        
        # 5. Penalización por sacudir bruscamente los controles
        delta = action - self.prev_action.astype(np.float64)
        r_smooth = -0.2 * float(np.dot(delta, delta))

        # Recompensa base
        reward = r_dist + r_upright + r_pendulum + r_hover + r_smooth
        
        # Súper-bono si logra colocar la carga exactamente en la meta
        if dist < 0.5:
            reward += 2.0
            
        return float(reward), False

    # ══════════════════════════════════════════════════════════════════════
    #  OBSERVACIÓN
    # ══════════════════════════════════════════════════════════════════════
    def _get_obs(self) -> np.ndarray:
        """
        IA/Aprendizaje: Prepara los sensores que verá la red neuronal.
        """
        obs = np.array([*self.state, self.x_sp, self.z_sp], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ══════════════════════════════════════════════════════════════════════
    #  RENDERIZADO (PYGAME)
    # ══════════════════════════════════════════════════════════════════════
    def _w2s(self, wx: float, wz: float):
        """Renderizado: Convierte coordenadas de metros físicos a píxeles de pantalla (invirtiendo el eje Z)."""
        px = int((wx - WORLD_X[0]) / (WORLD_X[1] - WORLD_X[0]) * SCREEN_W)
        py = int((1.0 - (wz - WORLD_Z[0]) / (WORLD_Z[1] - WORLD_Z[0])) * SCREEN_H)
        return px, py

    def _init_pygame(self):
        """Renderizado: Inicializa la ventana de Pygame y las fuentes de texto por única vez."""
        import pygame
        pygame.init()
        pygame.display.set_caption("Drone-Pendulum DRL  ·  Cuadricóptero 2D con Carga Suspendida")
        self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self._clock  = pygame.time.Clock()

        def _font(size, bold=False):
            try:
                return pygame.font.SysFont("Consolas", size, bold=bold)
            except Exception:
                return pygame.font.Font(None, size)

        self._font_sm = _font(15)
        self._font_md = _font(18, bold=True)
        self._font_lg = _font(22, bold=True)
        self._pg_init = True

    def render(self):
        """
        Renderizado: Dibuja toda la simulación frame por frame.
        Orden de dibujo (fondo al frente):
          1. Fondo oscuro y cuadrícula
          2. Suelo
          3. Marca del objetivo (setpoint)
          4. Cable y carga (péndulo)
          5. Cuerpo del dron y motores
          6. Flecha naranja de aceleración
          7. Flecha amarilla de viento
          8. Panel de telemetría (HUD)
          9. Barra de progreso inferior
        """
        import pygame

        if not self._pg_init:
            self._init_pygame()

        # ── Manejar eventos de cierre de ventana ───────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.close()
                sys.exit(0)

        scr = self._screen
        x, xd, z, zd, th, thd, ph, phd = self.state

        # ── 1. Fondo ───────────────────────────────────────────────────
        scr.fill(C_BG)

        # Líneas de la cuadrícula
        step_m = 2.0
        wx = WORLD_X[0]
        while wx <= WORLD_X[1]:
            gx, _ = self._w2s(wx, WORLD_Z[0])
            pygame.draw.line(scr, C_GRID, (gx, 0), (gx, SCREEN_H), 1)
            wx += step_m
        wz = WORLD_Z[0]
        while wz <= WORLD_Z[1]:
            _, gy = self._w2s(0, wz)
            pygame.draw.line(scr, C_GRID, (0, gy), (SCREEN_W, gy), 1)
            wz += step_m

        # ── 2. Línea de suelo ─────────────────────────────────────────
        _, gnd_y = self._w2s(0, 0)
        pygame.draw.line(scr, C_GROUND, (0, gnd_y), (SCREEN_W, gnd_y), 3)
        for gx in range(0, SCREEN_W, 18):
            pygame.draw.line(scr, (25, 55, 30), (gx, gnd_y), (gx - 12, gnd_y + 12), 1)

        # ── 3. Marca del setpoint ──────────────────────────────────────
        spx, spy = self._w2s(self.x_sp, self.z_sp)
        CR = 16
        pygame.draw.line(scr, C_SETPOINT, (spx - CR, spy), (spx + CR, spy), 2)
        pygame.draw.line(scr, C_SETPOINT, (spx, spy - CR), (spx, spy + CR), 2)
        pygame.draw.circle(scr, C_SETPOINT, (spx, spy), CR + 5, 1)
        for seg in range(12):
            ang0 = math.radians(seg * 30)
            ang1 = math.radians(seg * 30 + 15)
            pts = [
                (spx + int((CR + 12) * math.cos(a)),
                 spy + int((CR + 12) * math.sin(a)))
                for a in [ang0, ang1]
            ]
            pygame.draw.line(scr, (*C_SETPOINT[:3], 120), pts[0], pts[1], 1)
        lbl = self._font_sm.render("OBJETIVO", True, C_SETPOINT)
        scr.blit(lbl, (spx + CR + 8, spy - 8))

        # ── 4. Cable y carga (péndulo) ─────────────────────────────────
        drone_sx, drone_sy = self._w2s(x, z)
        bob_wx = x + l * math.sin(th)
        bob_wz = z - l * math.cos(th)
        bob_sx, bob_sy = self._w2s(bob_wx, bob_wz)

        pygame.draw.aaline(scr, C_CABLE, (drone_sx, drone_sy), (bob_sx, bob_sy))
        pygame.draw.line(scr, C_CABLE, (drone_sx, drone_sy), (bob_sx, bob_sy), 2)
        pygame.draw.circle(scr, (35, 100, 155), (bob_sx, bob_sy), 17)
        pygame.draw.circle(scr, C_BOB, (bob_sx, bob_sy), 13)
        pygame.draw.circle(scr, C_BOB_SHINE, (bob_sx - 4, bob_sy - 4), 4)

        # ── 5. Cuerpo del dron (rectángulo inclinado) ──────────────────
        HW = 50    # mitad del ancho en píxeles
        HH = 11    # mitad del alto en píxeles
        raw_corners = [(-HW, -HH), (HW, -HH), (HW, HH), (-HW, HH)]
        cos_ph, sin_ph = math.cos(ph), math.sin(ph)
        rot_corners = []
        for (px_, py_) in raw_corners:
            rx = px_ * cos_ph - py_ * sin_ph
            ry = px_ * sin_ph + py_ * cos_ph
            rot_corners.append((drone_sx + rx, drone_sy + ry))

        pygame.draw.polygon(scr, C_DRONE, rot_corners)
        pygame.draw.polygon(scr, C_DRONE_EDGE, rot_corners, 2)

        # Rotores en las esquinas
        for sign in (-1, +1):
            tip_local = (sign * (HW + 6), 0)
            rx = tip_local[0] * cos_ph - tip_local[1] * sin_ph
            ry = tip_local[0] * sin_ph + tip_local[1] * cos_ph
            rsx, rsy = int(drone_sx + rx), int(drone_sy + ry)
            pygame.draw.circle(scr, C_ROTOR, (rsx, rsy), 19, 2)
            pygame.draw.circle(scr, C_DRONE_EDGE, (rsx, rsy), 6)

        pygame.draw.circle(scr, C_COM, (drone_sx, drone_sy), 5)

        # ── 6. Flecha del vector de empuje ──────────────────────────────
        arrow_pix = 55    
        thrust_norm = self._last_uf / U_F_MAX
        t_len = int(arrow_pix * (0.3 + 0.7 * thrust_norm))
        up_x =  math.sin(ph)
        up_z  = math.cos(ph)
        t_end = (
            int(drone_sx + up_x * t_len),
            int(drone_sy - up_z * t_len),
        )
        pygame.draw.line(scr, C_THRUST_ARR, (drone_sx, drone_sy), t_end, 3)
        _draw_arrowhead(scr, C_THRUST_ARR, (drone_sx, drone_sy), t_end, size=9)

        # ── 7. Indicador de ráfaga de viento ────────────────────────────
        wf = self._wind_force
        if abs(wf) > 0.1:
            wf_frac   = wf / WIND_F_MAX
            arrow_len = int(abs(wf_frac) * 90)
            direction = 1 if wf > 0 else -1
            wy_scr    = 28
            wx_s      = SCREEN_W // 2 - 45
            wx_e      = wx_s + direction * arrow_len
            pygame.draw.line(scr, C_WIND_ARR, (wx_s, wy_scr), (wx_e, wy_scr), 4)
            _draw_arrowhead(scr, C_WIND_ARR, (wx_s, wy_scr), (wx_e, wy_scr), size=9)
            wlbl = self._font_sm.render(f"VIENTO {wf:+.1f} N", True, C_WIND_ARR)
            scr.blit(wlbl, (wx_s - 5, wy_scr + 10))

        # ── 8. Panel HUD ───────────────────────────────────────────────
        _draw_hud(scr, self._font_sm, self._font_md, self._font_lg,
                  self.state, self.x_sp, self.z_sp,
                  self.step_num, self._wind_force)

        # ── 9. Barra de progreso ───────────────────────────────────────
        bar_margin = 10
        bar_h      = 7
        bar_w      = SCREEN_W - 2 * bar_margin
        bar_y      = SCREEN_H - bar_h - 4
        prog       = self.step_num / MAX_STEPS
        pygame.draw.rect(scr, C_BAR_BG, (bar_margin, bar_y, bar_w, bar_h), border_radius=3)
        pygame.draw.rect(scr, C_BAR_FG,
                         (bar_margin, bar_y, int(bar_w * prog), bar_h),
                         border_radius=3)

        pygame.display.flip()
        self._clock.tick(FPS)

    # ══════════════════════════════════════════════════════════════════════
    #  CIERRE
    # ══════════════════════════════════════════════════════════════════════
    def close(self):
        """Ciclo de Vida: Libera recursos y cierra la ventana de Pygame."""
        if self._pg_init:
            import pygame
            pygame.quit()
            self._pg_init = False
            self._screen  = None


# ─────────────────────────────────────────────────────────────────────────────
#  FUNCIONES DE DIBUJO DE PYGAME
# ─────────────────────────────────────────────────────────────────────────────
def _draw_arrowhead(surf, color, start, end, size=10):
    """Renderizado: Dibuja la punta de una flecha (un triángulo relleno)."""
    import pygame
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist < 1:
        return
    ux, uy = dx / dist, dy / dist
    px, py = -uy, ux
    tip = end
    bl  = (end[0] - ux * size + px * size * 0.5,
           end[1] - uy * size + py * size * 0.5)
    br  = (end[0] - ux * size - px * size * 0.5,
           end[1] - uy * size - py * size * 0.5)
    pygame.draw.polygon(surf, color, [tip, bl, br])


def _draw_hud(surf, font_sm, font_md, font_lg,
              state, x_sp, z_sp, step_num, wind_force):
    """
    Renderizado: Dibuja el panel de instrumentos (HUD).
    Muestra los valores de física en tiempo real y pinta los errores en colores.
    """
    import pygame

    x, xd, z, zd, th, thd, ph, phd = state
    
    # Error medido desde la carga (péndulo) hacia el objetivo
    bob_x = x + l * math.sin(th)
    bob_z = z - l * math.cos(th)
    err_x = bob_x - x_sp
    err_z = bob_z - z_sp

    # Fondo semi-transparente del HUD
    HUD_X, HUD_Y = 12, 12
    HUD_W, HUD_H = 298, 358
    panel = pygame.Surface((HUD_W, HUD_H), pygame.SRCALPHA)
    panel.fill((*C_HUD_BG, 210))
    surf.blit(panel, (HUD_X, HUD_Y))
    pygame.draw.rect(surf, C_HUD_BORDER,
                     (HUD_X, HUD_Y, HUD_W, HUD_H), 1, border_radius=6)

    def _blit(text, color, x_off, y_off, font=None):
        f = font or font_sm
        s = f.render(text, True, color)
        surf.blit(s, (HUD_X + x_off, HUD_Y + y_off))

    def _err_color(val, scale=3.0):
        t = min(abs(val) / scale, 1.0)
        return (int(255 * t), int(255 * (1 - t)), 30)

    def _theta_color(val):
        t = min(abs(val) / (math.pi / 4), 1.0)
        return (int(255 * t), int(255 * (1 - t)), 0)

    DY = 22
    y  = 10
    _blit("DRON + PENDULO DRL",          C_HUD_TITLE, 14, y, font_md); y += DY + 4
    _blit("─" * 30,                      C_HUD_BORDER, 6, y);           y += 14
    _blit(f"  Paso   : {step_num:>5d} / {MAX_STEPS}", C_HUD_TXT, 6, y); y += DY
    _blit(f"  x      : {x:+8.3f} m",    C_HUD_TXT, 6, y);               y += DY
    _blit(f"  z      : {z:+8.3f} m",    C_HUD_TXT, 6, y);               y += DY
    _blit(f"  ẋ      : {xd:+8.3f} m/s", C_HUD_TXT, 6, y);               y += DY
    _blit(f"  ż      : {zd:+8.3f} m/s", C_HUD_TXT, 6, y);               y += DY
    _blit(f"  φ (Dron) : {math.degrees(ph):+8.2f} °",   C_HUD_TXT, 6, y); y += DY
    _blit(f"  θ (Pénd): {math.degrees(th):+8.2f} °",
          _theta_color(th), 6, y);                                        y += DY
    _blit(f"  θ̇      : {math.degrees(thd):+8.2f} °/s", C_HUD_TXT, 6, y);y += DY
    _blit("─" * 30,                      C_HUD_BORDER, 6, y);            y += 14
    _blit("  OBJETIVO (SETPOINT)",       C_HUD_SP, 6, y, font_md);       y += DY
    _blit(f"  x_sp   : {x_sp:+8.3f} m", C_HUD_SP, 6, y);                y += DY
    _blit(f"  z_sp   : {z_sp:+8.3f} m", C_HUD_SP, 6, y);                y += DY
    _blit("─" * 30,                      C_HUD_BORDER, 6, y);            y += 14
    _blit("  ERROR DE RASTREO",          C_HUD_ERR, 6, y, font_md);      y += DY
    _blit(f"  Δx     : {err_x:+8.3f} m", _err_color(err_x), 6, y);      y += DY
    _blit(f"  Δz     : {err_z:+8.3f} m", _err_color(err_z), 6, y);      y += DY
    _blit(f"  Dist   : {math.hypot(err_x, err_z):8.3f} m", C_HUD_TXT, 6, y); y += DY
    _blit(f"  Viento : {wind_force:+8.2f} N",
          C_WIND_ARR if abs(wind_force) > 0.1 else C_HUD_TXT, 6, y)