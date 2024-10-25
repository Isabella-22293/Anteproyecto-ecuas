import numpy as np
import matplotlib.pyplot as plt

def runge_kutta_4(f, y0, t0, tn, h):
    n_steps = int((tn - t0) / h)
    t_values = np.linspace(t0, tn, n_steps + 1)
    y_values = np.zeros((n_steps + 1,) + np.shape(y0))
    y_values[0] = y0

    for n in range(n_steps):
        k1 = h * f(t_values[n], y_values[n])
        k2 = h * f(t_values[n] + h / 2, y_values[n] + k1 / 2)
        k3 = h * f(t_values[n] + h / 2, y_values[n] + k2 / 2)
        k4 = h * f(t_values[n] + h, y_values[n] + k3)
        y_values[n + 1] = y_values[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return t_values, y_values

def adams_bashforth_moulton(f, y0, t0, tn, h):
    n_steps = int((tn - t0) / h)
    t_values = np.linspace(t0, tn, n_steps + 1)
    y_values = np.zeros((n_steps + 1,) + np.shape(y0))
    y_values[0] = y0

    # Usar RK4 para el primer paso
    t_temp, y_temp = runge_kutta_4(f, y0, t0, t0 + h, h)
    y_values[1] = y_temp[1]

    for n in range(1, n_steps):
        predictor = y_values[n] + (h / 2) * (f(t_values[n], y_values[n]) + f(t_values[n - 1], y_values[n - 1]))
        y_values[n + 1] = y_values[n] + (h / 2) * (f(t_values[n + 1], predictor) + f(t_values[n], y_values[n]))

    return t_values, y_values

# EDO de Primer Orden
def f1(t, y):
    return -2 * y + 3 * np.exp(-t)

# EDO de Segundo Orden
def f2(t, y):
    return np.array([y[1], -y[0]])

# Sistema de EDOs
def f3(t, y):
    return np.array([y[0] + 2 * y[1], -y[0] + y[1]])

# Seleccionar tipo de EDO
edo_type = int(input("Selecciona la ecuación a resolver (1: Primer Orden, 2: Segundo Orden, 3: Sistema 2x2): "))

# Definir parámetros comunes
t0 = 0
tn_1 = 5  # Tiempo final para EDO de primer orden y sistema
tn_2 = 10  # Tiempo final para EDO de segundo orden
h = 0.1

if edo_type == 1:
    # Resolver EDO de Primer Orden
    y0_1 = np.array([1])
    t_values_rk4_1, y_values_rk4_1 = runge_kutta_4(f1, y0_1, t0, tn_1, h)
    t_values_abm_1, y_values_abm_1 = adams_bashforth_moulton(f1, y0_1, t0, tn_1, h)

    # Solución analítica
    y_analytical_1 = (3 / 2) - (1 / 2) * np.exp(-2 * t_values_rk4_1)

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(t_values_rk4_1, y_values_rk4_1, label='RK4 (Numérico)', color='blue')
    plt.plot(t_values_abm_1, y_values_abm_1, label='ABM (Numérico)', color='green')
    plt.plot(t_values_rk4_1, y_analytical_1, label='Solución Analítica', color='red', linestyle='dashed')
    plt.title('EDO de Primer Orden')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('Solución (y)')
    plt.legend()
    plt.grid()
    plt.show()

elif edo_type == 2:
    # Resolver EDO de Segundo Orden
    y0_2 = np.array([1, 0])  # [y(0), y'(0)]
    t_values_rk4_2, y_values_rk4_2 = runge_kutta_4(f2, y0_2, t0, tn_2, h)
    t_values_abm_2, y_values_abm_2 = adams_bashforth_moulton(f2, y0_2, t0, tn_2, h)

    # Solución analítica para EDO de Segundo Orden
    y_analytical_2 = np.array([np.cos(t_values_rk4_2), np.sin(t_values_rk4_2)])  # [y, y']

    # Graficar resultados EDO de Segundo Orden
    plt.figure(figsize=(10, 6))

    # Graficar la primera variable (y1)
    plt.plot(t_values_rk4_2, y_values_rk4_2[:, 0], label='RK4 (Numérico) y1', color='blue')
    plt.plot(t_values_abm_2, y_values_abm_2[:, 0], label='ABM (Numérico) y1', color='green')
    plt.plot(t_values_rk4_2, y_analytical_2[0], label='Solución Analítica y1', color='red', linestyle='dashed')

    # Graficar la segunda variable (y2)
    plt.plot(t_values_rk4_2, y_values_rk4_2[:, 1], label='RK4 (Numérico) y2', color='orange')
    plt.plot(t_values_abm_2, y_values_abm_2[:, 1], label='ABM (Numérico) y2', color='purple')
    plt.plot(t_values_rk4_2, y_analytical_2[1], label='Solución Analítica y2', color='pink', linestyle='dashed')

    plt.title('EDO de Segundo Orden')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('Solución')
    plt.legend()
    plt.grid()
    plt.show()


elif edo_type == 3:
    # Resolver Sistema de EDOs
    y0_3 = np.array([1, 0])  # [y1(0), y2(0)]
    t_values_rk4_3, y_values_rk4_3 = runge_kutta_4(f3, y0_3, t0, tn_1, h)
    t_values_abm_3, y_values_abm_3 = adams_bashforth_moulton(f3, y0_3, t0, tn_1, h)

    # Solución analítica para Sistema de EDOs
    y_analytical1 = np.exp(t_values_rk4_3)  # y1(t)
    y_analytical2 = y_analytical1 - 1        # y2(t)

    # Graficar resultados Sistema de EDOs
    plt.figure(figsize=(10, 6))
    
    # Graficar la primera variable (y1)
    plt.plot(t_values_rk4_3, y_values_rk4_3[:, 0], label='RK4 (Numérico) y1', color='blue')
    plt.plot(t_values_abm_3, y_values_abm_3[:, 0], label='ABM (Numérico) y1', color='green')
    plt.plot(t_values_rk4_3, y_analytical1, label='Solución Analítica y1', color='red', linestyle='dashed')

    # Graficar la segunda variable (y2)
    plt.plot(t_values_rk4_3, y_values_rk4_3[:, 1], label='RK4 (Numérico) y2', color='orange')
    plt.plot(t_values_abm_3, y_values_abm_3[:, 1], label='ABM (Numérico) y2', color='purple')
    plt.plot(t_values_rk4_3, y_analytical2, label='Solución Analítica y2', color='pink', linestyle='dashed')

    plt.title('Sistema de Ecuaciones Diferenciales de 2x2')
    plt.xlabel('Tiempo (t)')
    plt.ylabel('Solución')
    plt.legend()
    plt.grid()
    plt.show()
