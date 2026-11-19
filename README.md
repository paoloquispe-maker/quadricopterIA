# Drone-Pendulum DRL — Quadricóptero 2D con Carga Suspendida

Un entorno de **Gymnasium** personalizado para entrenar a un agente de Aprendizaje por Refuerzo Profundo (DRL) utilizando PPO para controlar un quadricóptero 2D que transporta una carga de péndulo suspendida.

---

## Descripción General del Sistema
```
       ┌────────────────────────────────────┐
       │          Plano X-Z (2D)            │
       │                                    │
       │        u_f  ↑   u_τ  ↻             │
       │          ╔══╪══╗                   │
       │          ║Drone║  ←— φ (roll)      │
       │          ╚══╪══╝                   │
       │             │   Cable (l = 1.0 m)  │
       │             ●   Carga (m = 0.5 kg) │
       │             ↕   θ (ángulo péndulo) │
       │                                    │
       └────────────────────────────────────┘
```

## Física

El sistema está descrito por **8 variables de estado**:

| Símbolo | Descripción |
|--------|-------------|
| `x` | Posición horizontal del drone (m) |
| `ẋ` | Velocidad horizontal del drone (m/s) |
| `z` | Altura del drone (m) |
| `ż` | Velocidad vertical del drone (m/s) |
| `θ` | Ángulo del péndulo desde la vertical descendente (rad) |
| `θ̇` | Velocidad angular del péndulo (rad/s) |
| `φ` | Ángulo de alabeo del drone (rad) |
| `φ̇` | Velocidad angular de alabeo del drone (rad/s) |

Las ecuaciones de movimiento se derivan usando **Euler-Lagrange** y se resuelven en cada paso de tiempo usando `numpy.linalg.solve` para la matriz de masa acoplada 3×3, luego se integran con Euler explícito en `Δt = 0.02 s`.

## Inicio Rápido

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el agente (100 000 pasos)
```bash
python train.py
```
Los modelos se guardan en `./models/`. Los logs de TensorBoard van a `./logs/`.

### 3. Observar el agente entrenado
```bash
python enjoy.py
```
Se abre una ventana de matplotlib con telemetría en vivo.

## Estructura de Archivos

```
drone_pendulum_drl/
├── drone_env.py      # Clase de entorno Gymnasium (física + renderizado)
├── train.py          # Script de entrenamiento PPO (SB3)
├── enjoy.py          # Script de inferencia + visualización
├── requirements.txt  # Dependencias de Python
└── README.md         # Este archivo
```

## Función de Recompensa

| Componente | Peso | Descripción |
|-----------|--------|-------------|
| Proximity bonus | +2.0 | Recompensa Gaussiana por estar cerca del setpoint |
| Tracking error (x) | −2.0 | Penaliza el desplazamiento horizontal |
| Tracking error (z) | −2.0 | Penaliza el desplazamiento de altitud |
| Pendulum swing (θ²) | −5.0 | Objetivo principal de estabilidad |
| Pendulum rate (θ̇²) | −0.5 | Amortigua las oscilaciones |
| Roll angle (φ²) | −0.5 | Mantiene el drone nivelado |
| Thrust deviation | −0.005 | Fomenta un empuje nivelado de hover |
| Action smoothness | −0.2 | Penaliza cambios abruptos de control |
| Velocity damping | −0.1 | Ralentiza el movimiento innecesario |
| **Crash penalty** | **−150** | Termina el episodio |

## Constantes Físicas

| Constante | Valor | Unidad |
|----------|-------|------|
| Drone mass M | 1.5 | kg |
| Load mass m | 0.5 | kg |
| Cable length l | 1.0 | m |
| Gravity g | 9.81 | m/s² |
| Roll inertia I | 0.02 | kg·m² |
| Max thrust | 40.0 | N |
| Max torque | ±2.0 | N·m |
