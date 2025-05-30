
import numpy as np
import matplotlib.pyplot as plt



"""
Sección 7.1

Ejercicio 3:

3. Integrar


y' = sin y  y(0) = 1

desde x = 0 hasta x = 0.5 usando el método de Euler 
con h = 0.1. Compara el resultado con el Ejemplo 7.3.
"""



# ---------- Método de Euler ----------
def eulerint(F, x, y, xStop, h):
    X = [x]
    Y = [[y]]      # <- Aquí, lista dentro de lista para que sea 2D
    while x < xStop:
        h = min(h, xStop - x)
        y = y + h * F(x, y)
        x = x + h
        X.append(x)
        Y.append([y])  # <- Aquí también, agrega la lista
    return np.array(X), np.array(Y)


# ---------- Imprimir solución ----------
def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n x ", end=" ")
        for i in range(n):
            print(" y[", i, "] ", end=" ")
        print()
    def imprimeLinea(x, y, n):
        print("{:13.4e}".format(x), end=" ")
        for i in range(n):
            print("{:13.4e}".format(y[i]), end=" ")
        print()
    try:
        n = len(Y[0])
    except TypeError:
        n = 1
    if frec == 0:
        frec = len(X)
    imprimeEncabezado(n)
    for i in range(0, len(X), frec):
        imprimeLinea(X[i], Y[i], n)
    if i != len(X) - 1:
        imprimeLinea(X[-1], Y[-1], n)

# ---------- Sistema de EDOs ----------
def F(x, y):
    return np.sin(y)

y0 = 1.0  # y(0) = 1

# ---------- Solución analítica para ángulos pequeños ----------
def yExacta(x):
    A = np.tan(0.5)
    return 2 * np.arctan(A * np.exp(x))

# ---------- Ejecutar método de Euler ----------
X, Y = eulerint(F, 0.0, y0, 0.5, 0.1)

# ---------- Mostrar resultados ----------
print("La solución es")
imprimeSol(X, Y, 4)

# ---------- Calcular errores ----------
print("\nEl error es")
Eac = 0.0
print("   x       y    yExacta  Error (%)  Error acumulado")
print("-----------------------------------------------------")
print(f"{X[0]:.4f}  {Y[0,0]:.4f}  {yExacta(X[0]):.4f}  {0.0:.4f}  {Eac:.4f}")
for i in range(1, len(X)):
    exacta = yExacta(X[i])
    E = abs((Y[i, 0] - exacta) / exacta) * 100
    Eac += E
    print(f"{X[i]:.4f}  {Y[i,0]:.4f}  {exacta:.4f}  {E:.4f}  {Eac:.4f}")

# ---------- Gráfica ----------
plt.plot(X, yExacta(X), label="Solución Exacta", linestyle="--", color="orange")
plt.plot(X, Y[:, 0], label="Método de Euler", linewidth=2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo θ (rad)")
plt.title("Comparación entre solución exacta y método de Euler")
plt.grid()
plt.legend()
plt.show()

#El resultado es muy similar al ejemplo 7.3
#%%%%%%%%

"""
4. Verificar que el problema

y' = y^(1/3),  y(0) = 0

tiene dos soluciones:
1. y = 0
2. y = (2x/3)^(3/2)

¿Cuál de las soluciones se reproducirá mediante integración numérica si la condición inicial es:
a) y = 0
b) y = 1e-16?

Verificar las conclusiones integrando con algún método numérico.
"""

#En este caso nos sirve usar el metodo de Euler para comparar 
#cual de las dos soluciones analíticas serán parecidas a la integración
#numerica con distintas condiciones iniciales

#Para la solución 1 del inciso a)
# ---------- Integracion numerica para el inciso a) con el valor inicial de 0 ----------
def F(x, y):
    return y**(1/3)

y0 = 0.0  # y(0) = 0

# ---------- Solución analítica y = 0 ----------
def yExacta(x):
    return np.zeros_like(x)# es una función de NumPy que crea un array de ceros con la misma forma (dimensión y tamaño) que el array x.
#Se hace para poder graficar por que si no solo tendría un solo punto


# ---------- Ejecutar método de Euler ----------
X, Y = eulerint(F, 0.0, y0, 10, 0.1)

# ---------- Mostrar resultados ----------
print("La solución es")
imprimeSol(X, Y, 4)

# ---------- Calcular errores ----------
print("\nEl error es")
Eac = 0.0
print("   x       y    yExacta  Error (%)  Error acumulado")
print("-----------------------------------------------------")
print(f"{X[0]:.4f}  {Y[0,0]:.4f}  {yExacta(X[0]):.4f}  {0.0:.4f}  {Eac:.4f}")
for i in range(1, len(X)):
    exacta = yExacta(X[i])
    yi = Y[i, 0]
    if exacta != 0:
        E = abs((yi - exacta) / exacta) * 100
    else:
        E = abs(yi - exacta)  # Esta parte es para evitar la division entre cero
    Eac += E
    print(f"{X[i]:.4f}  {yi:.4f}  {exacta:.4f}  {E:.4f}  {Eac:.4f}")

# ---------- Gráfica ----------
plt.plot(X, yExacta(X),  '--', label="Solución exacta: y = 0", color="orange",linewidth = 2)
plt.plot(X, Y[:, 0], label="Método de Euler", linewidth=2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo θ (rad)")
plt.title("Comparación entre solución exacta y = 0 y método de Euler, con el valor inicial y = 0")
plt.grid()
plt.legend()
plt.show()

#Para la solución 1 del inciso b)
# ---------- Integracion numerica para el inciso b) con el valor inicial de 1e-16 ----------
def F(x, y):
    return y**(1/3)

y0 = 1e-16  # y(0) = 1e-16

# ---------- Solución analítica y = 0 ----------
def yExacta(x):
    return np.zeros_like(x)# es una función de NumPy que crea un array de ceros con la misma forma (dimensión y tamaño) que el array x.
#Se hace para poder graficar por que si no solo tendría un solo punto


# ---------- Ejecutar método de Euler ----------
X, Y = eulerint(F, 0.0, y0, 10, 0.1)

# ---------- Mostrar resultados ----------
print("La solución es")
imprimeSol(X, Y, 4)

# ---------- Calcular errores ----------
print("\nEl error es")
Eac = 0.0
print("   x       y    yExacta  Error (%)  Error acumulado")
print("-----------------------------------------------------")
print(f"{X[0]:.4f}  {Y[0,0]:.4f}  {yExacta(X[0]):.4f}  {0.0:.4f}  {Eac:.4f}")
for i in range(1, len(X)):
    exacta = yExacta(X[i])
    yi = Y[i, 0]
    if exacta != 0:
        E = abs((yi - exacta) / exacta) * 100
    else:
        E = abs(yi - exacta)  # Esta parte es para evitar la division entre cero
    Eac += E
    print(f"{X[i]:.4f}  {yi:.4f}  {exacta:.4f}  {E:.4f}  {Eac:.4f}")

# ---------- Gráfica ----------
plt.plot(X, yExacta(X),  '--', label="Solución exacta: y = 0", color="orange",linewidth = 2)
plt.plot(X, Y[:, 0], label="Método de Euler", linewidth=2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo θ (rad)")
plt.title("Comparación entre solución exacta y = 0 y método de Euler, con el valor inicial y = 1e-16")
plt.grid()
plt.legend()
plt.show()
#$%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#Para la solucion 2  del inciso a)
# ---------- Integracion numerica para el inciso a) con el valor inicial de 0 ----------
def F(x, y):
    return y**(1/3)

y0 = 0.0  # y(0) = 0

#---------- Solución analítica y =(2x/3)^(3/2) ----------
def yExacta(x):
    return (2*x/3)**(3/2)


# ---------- Ejecutar método de Euler ----------
X, Y = eulerint(F, 0.0, y0, 10, 0.1)

# ---------- Mostrar resultados ----------
print("La solución es")
imprimeSol(X, Y, 4)

# ---------- Calcular errores ----------
print("\nEl error es")
Eac = 0.0
print("   x       y    yExacta  Error (%)  Error acumulado")
print("-----------------------------------------------------")
print(f"{X[0]:.4f}  {Y[0,0]:.4f}  {yExacta(X[0]):.4f}  {0.0:.4f}  {Eac:.4f}")
for i in range(1, len(X)):
    exacta = yExacta(X[i])
    E = abs((Y[i, 0] - exacta) / exacta) * 100
    Eac += E
    print(f"{X[i]:.4f}  {Y[i,0]:.4f}  {exacta:.4f}  {E:.4f}  {Eac:.4f}")

# ---------- Gráfica ----------
plt.plot(X, yExacta(X),  '--', label="Solución exacta: y = (2x/3)^(3/2)", color="orange")
plt.plot(X, Y[:, 0], label="Método de Euler", linewidth=2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo θ (rad)")
plt.title("Comparación entre solución exacta y =(2x/3)^(3/2) y método de Euler, con el valor inicial y = 0.")
plt.grid()
plt.legend()
plt.show()

#Para la solucion 2  del inciso b)
# ---------- Integracion numerica para el inciso a) con el valor inicial de 1e-16 ----------
def F(x, y):
    return y**(1/3)

y0 = 1e-16  # y(0) = 1e-16

#---------- Solución analítica y =(2x/3)^(3/2) ----------
def yExacta(x):
    return (2*x/3)**(3/2)


# ---------- Ejecutar método de Euler ----------
X, Y = eulerint(F, 0.0, y0, 10, 0.1)

# ---------- Mostrar resultados ----------
print("La solución es")
imprimeSol(X, Y, 4)

# ---------- Calcular errores ----------
print("\nEl error es")
Eac = 0.0
print("   x       y    yExacta  Error (%)  Error acumulado")
print("-----------------------------------------------------")
print(f"{X[0]:.4f}  {Y[0,0]:.4f}  {yExacta(X[0]):.4f}  {0.0:.4f}  {Eac:.4f}")
for i in range(1, len(X)):
    exacta = yExacta(X[i])
    E = abs((Y[i, 0] - exacta) / exacta) * 100
    Eac += E
    print(f"{X[i]:.4f}  {Y[i,0]:.4f}  {exacta:.4f}  {E:.4f}  {Eac:.4f}")

# ---------- Gráfica ----------
plt.plot(X, yExacta(X),  '--', label="Solución exacta: y = (2x/3)^(3/2)", color="orange")
plt.plot(X, Y[:, 0], label="Método de Euler", linewidth=2)
plt.xlabel("Tiempo (s)")
plt.ylabel("Ángulo θ (rad)")
plt.title("Comparación entre solución exacta y =(2x/3)^(3/2) y método de Euler, con el valor inicial y = 1e-16.")
plt.grid()
plt.legend()
plt.show()




#Entonces la solucion numerica seguirá la solución numérica seguirá la solución no trivial y = (2x/3)^(3/2)
#La solución 2 se reproducirá mediante integración númerica con el valor inicial 1e-16

#%%%%%%%%%
"""
3. Bosqueja aproximadamente la solución de los siguientes 
problemas con condiciones de frontera.
Usa el bosquejo para estimar y'(0) para cada problema.

(a) y'' = -e^{-y},   y(0) = 1,   y(1) = 0.5

(b) y'' = 4y^2,      y(0) = 10,  y'(1) = 0
"""
# Método de disparo para y'' = -e^{-y}, con y(0) = 1 y y(1) = 0.5

import matplotlib.pyplot as plt

#############   INCISO A)
# ------------ Define el sistema de ecuaciones 1er orden ------------
def F1(x, y):
    F = np.zeros(2)
    F[0] = y[1]                 # y0' = y1
    F[1] = -np.exp(-y[0])       # y1' = -e^{-y0} = -e^{-y}
    return F

# ------------ Condición inicial: y(0) = 1, y'(0) = u ------------
def initCond(u): 
    return np.array([1.0, u])   # y(0) = 1 (fijo), y'(0) = u (a encontrar)

# ------------ Función objetivo para Ridder (buscar u) ------------
def r(u):
    X, Y = Run_Kut4(F1, x, initCond(u), xStop, h)
    y = Y[-1]                   # Último valor de y(xStop)
    r = y[0] - 0.5              # Queremos que y(1) = 0.5 → r(u) = y - 0.5
    return r

# ------------ Método RK4 ------------
def Run_Kut4(F, x, y, xStop, h):
    def run_kut4(F, x, y, h):
        K0 = h*F(x, y)
        K1 = h*F(x + h/2.0, y + K0/2.0)
        K2 = h*F(x + h/2.0, y + K1/2.0)
        K3 = h*F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3)/6.0

    X = [x]
    Y = [y]
    while x < xStop:
        h = min(h, xStop - x)
        y = y + run_kut4(F, x, y, h)
        x = x + h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# ------------ Imprimir solución ------------
def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n x ", end=" ")
        for i in range(n):
            print(" y[", i, "] ", end=" ")
        print()

    def imprimeLinea(x, y, n):
        print("{:13.4e}".format(x), end=" ")
        for i in range(n):
            print("{:13.4e}".format(y[i]), end=" ")
        print()

    m = len(Y)
    try: n = len(Y[0])
    except TypeError: n = 1
    if frec == 0: frec = m
    imprimeEncabezado(n)
    for i in range(0, m, frec):
        imprimeLinea(X[i], Y[i], n)
    if i != m - 1: imprimeLinea(X[m - 1], Y[m - 1], n)

# ------------ Método de Ridder para encontrar la pendiente inicial u ------------
def Ridder(f, a, b, tol=1.0e-9):
    fa = f(a)
    if fa == 0.0: return a
    fb = f(b)
    if fb == 0.0: return b
    if np.sign(fa) != np.sign(fb): c = a
    for i in range(30):
        c = 0.5*(a + b)
        fc = f(c)
        s = np.sqrt(fc**2 - fa*fb)
        if s == 0.0: return None
        dx = (c - a)*fc/s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx
        fx = f(x)
        if i > 0 and abs(x - xOld) < tol * max(abs(x), 1.0):
            return x
        xOld = x
        if np.sign(fc) == np.sign(fx):
            if np.sign(fa) != np.sign(fx):
                b = x; fb = fx
            else:
                a = x; fa = fx
        else:
            a = c; b = x; fa = fc; fb = fx
    return None

# ------------ Parámetros del problema ------------
x, xStop = 0.0, 1.0  # Dominio de integración
h = 0.1              # Paso

# ------------ Ejecutar método de disparo ------------
u = Ridder(r, -10.0, 10.0)  # Buscar valor inicial y'(0) entre -10 y 10
X, Y = Run_Kut4(F1, x, initCond(u), xStop, h)

# ------------ Mostrar resultados ------------
print("La solución es")
imprimeSol(X, Y, 2)

# ------------ Graficar ------------
plt.plot(X, Y[:, 0], "o-", label="Método de disparo")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.title("Solución del BVP: y'' = -e^{-y}")
plt.legend()
plt.grid()
plt.show()

###### INCISO B)


# --- Función del sistema: y'' = 4y^2 convertido a sistema de primer orden
def F2(x, y):
    F = np.zeros(2)
    F[0] = y[1]
    F[1] = 4.0 * y[0]**2
    return F

# --- Condiciones iniciales
def initCond_b(u):
    return np.array([10.0, u])

# --- Método RK4
def Run_Kut4(F, x, y, xStop, h):
    def run_kut4(F, x, y, h):
        K0 = h * F(x, y)
        K1 = h * F(x + h/2.0, y + K0/2.0)
        K2 = h * F(x + h/2.0, y + K1/2.0)
        K3 = h * F(x + h, y + K2)
        return (K0 + 2.0*K1 + 2.0*K2 + K3) / 6.0

    X = [x]
    Y = [y]
    while x < xStop:
        h = min(h, xStop - x)
        y = y + run_kut4(F, x, y, h)
        x += h
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

# --- Función r(u)
def r_b(u):
    try:
        X, Y = Run_Kut4(F2, 0.0, initCond_b(u), 1.0, 0.1)
        return Y[-1][1]
    except:
        return np.nan

# --- Método de Ridder
def Ridder(f, a, b, tol=1.0e-9): 
    fa = f(a)
    fb = f(b)
    if np.isnan(fa) or np.isnan(fb):
        raise ValueError("Intervalo inválido, r_b(u) es NaN en los extremos")
    if fa == 0.0: return a
    if fb == 0.0: return b
    if np.sign(fa) == np.sign(fb):
        raise ValueError("f(a) y f(b) deben tener signos opuestos")
    for i in range(30):
        c = 0.5*(a + b)
        fc = f(c)
        s = np.sqrt(fc**2 - fa*fb)
        if s == 0.0: return None
        dx = (c - a)*fc/s
        if (fa - fb) < 0.0: dx = -dx
        x = c + dx
        fx = f(x)
        if i > 0 and abs(x - xOld) < tol * max(abs(x), 1.0):
            return x
        xOld = x
        if np.sign(fc) == np.sign(fx):
            if np.sign(fa) != np.sign(fx): b = x; fb = fx
            else: a = x; fa = fx
        else:
            a = c; b = x; fa = fc; fb = fx
    print("Demasiadas iteraciones")
    return None

# --- Imprimir solución
def imprimeSol(X, Y, frec):
    def imprimeEncabezado(n):
        print("\n x ", end=" ")
        for i in range(n):
            print(f" y[{i}] ", end=" ")
        print()

    def imprimeLinea(x, y, n):
        print(f"{x:13.4e}", end=" ")
        for i in range(n):
            print(f"{y[i]:13.4e}", end=" ")
        print()

    m = len(Y)
    try: n = len(Y[0])
    except TypeError: n = 1
    if frec == 0: frec = m
    imprimeEncabezado(n)
    for i in range(0, m, frec):
        imprimeLinea(X[i], Y[i], n)
    if i != m - 1:
        imprimeLinea(X[-1], Y[-1], n)
        

x = 0.0
xStop = 1.0
h = 0.1

X, Y = Run_Kut4(F2, x, initCond_b(u), xStop, h)

print("La solución es:")
imprimeSol(X, Y, 2)

import matplotlib.pyplot as plt
plt.plot(X, Y[:, 0], label="y(x)")
plt.plot(X, Y[:, 1], label="y'(x)")
plt.xlabel("x")
plt.ylabel("y, y'")
plt.title("Solución del BVP: y'' = 4 y^2")
plt.grid()
plt.legend()
plt.show()
#a partir de xStop = 0.7 la  funcion explota a numeros extremadamente numeros muy altos