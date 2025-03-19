import numpy as np
import matplotlib.pyplot as plt

#Ejercicio Numero 1
"""
Ejercicio:

Dada la siguiente tabla de valores de temperatura T y viscosidad cinética μ_k del agua:

T (°C)    μ_k (10⁻³ m²/s)
---------------------------------
0         0.101
21.1      1.79
37.8      1.13
54.4      0.696
71.1      0.519
87.8      0.338
100       0.296

Utiliza un método de interpolación en Python para estimar los valores de μ_k en T = 10°C, 30°C, 60°C y 90°C.

Luego, representa gráficamente los datos originales y los valores interpolados en una misma gráfica.
"""


xData = np.array([0, 21.1, 37.8, 54.4, 71.1, 87.8, 100])  # Temperaturas (T en °C)
yData = np.array([0.101, 1.79, 1.13, 0.696, 0.519, 0.338, 0.296])  # Viscosidad (μ_k)


xInterp = np.array([10, 30, 60, 90])# Puntos a interpolar

# Método de interpolación de Newton
def coeffts(xData, yData):
    m = len(xData)  # Número de datos
    a = yData.copy()  # Copia los valores de viscosidad (yData)
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (xData[k:m] - xData[k - 1])  # Cálculo de las diferencias divididas
    return a

def evalPoly(a, xData, x):  # Función que evalúa el polinomio de Newton
    n = len(xData) - 1  # Grado del polinomio
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - xData[n - k]) * p
    return p

# Obtener los coeficientes de interpolación de Newton
coeff = coeffts(xData, yData)

# Evaluar el polinomio de Newton en los puntos de interpolación
yInterp = [evalPoly(coeff, xData, x) for x in xInterp]

# Resultados de la interpolación
print("xInterpolado - yInterpolado")
print("--------------------------------")
for i in range(len(xInterp)):
    print(" %.1f  %.8f" % (xInterp[i], yInterp[i]))

# Graficar los resultados
xFine = np.linspace(min(xData), max(xData), 100)  # Valores finos de x para graficar la curva suavizada
yFine = [evalPoly(coeff, xData, x) for x in xFine]  # Evaluamos el polinomio para estos valores

plt.plot(xFine, yFine, label="Polinomio de Newton", color='r')
plt.plot(xInterp, yInterp, "x", label="Puntos interpolados", color='b')
plt.plot(xData, yData, "o", label="Datos")
plt.xlabel("Temperaturas (T en °C)")
plt.ylabel("Viscosidad [μ_k(10^-3m^2/s)]")
plt.legend()
plt.grid()
plt.show()

#Ejercicio 2
"""
La tabla muestra como la densidad relativa ρ del aire varía con la altura h. Determine mediante interpolación de Lagrange la densidad relativa del aire a 10.5 km.

h (km)      ρ  
0               1  
1.525       0.8617  
3.050       0.7385  
4.575       0.6292  
6.10         0.5328  
7.625       0.4481  
9.150       0.3741  
"""


import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


x_points = np.array([0, 1.525, 3.050, 4.575, 6.10, 7.625, 9.150])  # Temperaturas (T en °C)
y_points = np.array([1, 0.8617, 0.7385, 0.6292, 0.5328, 0.4481, 0.3741])  # Viscosidad (μ_k)
xp = 10.5




def lagrange_1(x_points, y_points, xp):
    """
    Calcula y grafica el polinomio de interpolación de Lagrange.

    Parámetros:
    x_points (list or array): Puntos en el eje x.
    y_points (list or array): Puntos en el eje y.
    xp (float): Punto en el que se desea interpolar.

    Retorna:
    yp (float): Valor interpolado en xp.
    """
    m = len(x_points)
    n = m - 1
    # Definir la variable simbólica
    x = sp.symbols("x")

    # Función para calcular los polinomios básicos de Lagrange
    def lagrange_basis(xp, x_points, i):
        L_i = 1
        for j in range(len(x_points)):
            if j != i:
                L_i *= (xp - x_points[j]) / (x_points[i] - x_points[j])
        return L_i

    # Función para calcular el polinomio de Lagrange
    def lagrange_interpolation(xp, x_points, y_points):
        yp = 0
        for i in range(len(x_points)):
            yp += y_points[i] * lagrange_basis(xp, x_points, i)
        return yp

    # Calcular el valor interpolado
    yp = lagrange_interpolation(xp, x_points, y_points)
    print("For x = %.1f, y = %.1f" % (xp, yp))

    # Crear puntos para la interpolación
    x_interpolado = np.linspace(min(x_points), max(x_points), 100)
    y_interpolado = [
        lagrange_interpolation(x_val, x_points, y_points) for x_val in x_interpolado
    ]

    # Graficar los puntos originales
    plt.scatter(x_points, y_points, label="Puntos Originales", color="red")

    # Graficar el polinomio de interpolación de Lagrange
    plt.plot(
        x_interpolado, y_interpolado, label="Interpolación de Lagrange", linestyle="-"
    )

    # Graficar el valor interpolado
    plt.scatter(xp, yp, label="Punto interpolado", color="blue", zorder=5)
    plt.text(xp, yp, f"({xp:.1f}, {yp:.1f})", fontsize=12, verticalalignment="bottom")

    # Añadir etiquetas y leyenda
    plt.xlabel("h(km)")
    plt.ylabel("Densidad Relativa")
    plt.title("Polinomio de Interpolación de Lagrange")
    plt.legend()
    plt.grid(True)

    # Mostrar la gráfica
    plt.show()

    # Construir el polinomio de interpolación simbólicamente
    polinomio = 0
    for i in range(len(x_points)):
        term = y_points[i]
        for j in range(len(x_points)):
            if j != i:
                term *= (x - x_points[j]) / (x_points[i] - x_points[j])
        polinomio += term

    # Simplificar el polinomio
    polinomio_simplificado = sp.simplify(polinomio)

    # Imprimir el polinomio de interpolación
    print("Polinomio de Interpolación de Lagrange:")
    print(f"y(x) = {polinomio}")
    print("\nPolinomio Simplificado:")
    print(f"y(x) = {polinomio_simplificado}")

    return yp

lagrange_1(x_points, y_points, xp)

#Ejercicio 3
"""
    
    3. La amplitud vibracional de un eje de transmisión es medida a varias velocidades. Los resultados son

Velocidad (rpm)   Amplitud (mm)  
0                          0  
400                      0.072  
800                      0.233  
1200                    0.712  
1600                    3.400  

Utilice el método de interpolación más conveniente para graficar amplitud vs velocidad de 0 a 2500 rpm (observe los intervalos de la tabla y determine el tamaño más conveniente de los intervalos).
"""
#Dato originales
xData = np.array([0, 400 , 800, 1200, 1600])  # Velocidad (rpm)
yData = np.array([0, 0.072, 0.233, 0.712, 3.400])  # Amplitud (mm)
    
# Método de interpolación de Newton


def evalPoly(a, xData, x):  # Función que evalua polinomios de Lagrange
    n = len(xData) - 1  # Grado del polinomio
    p = a[n]
    for k in range(1, n + 1):
        p = a[n - k] + (x - xData[n - k]) * p
    return p


def coeffts(xData, yData):
    m = len(xData)  # Número de datos
    a = yData.copy()
    for k in range(1, m):
        a[k:m] = (a[k:m] - a[k - 1]) / (xData[k:m] - xData[k - 1])
    return a


coeff = coeffts(xData, yData)

# para que el polinomio se vea mas fluido y suave
Xint = np.linspace(0, 2500, 100)
yInt = np.array([evalPoly(coeff, xData, xi) for xi in Xint])

# rango para los valor y puntos a interpolar
Xinterp = np.arange(0, 2500, 250)
yInterp = np.array([evalPoly(coeff, xData, xi) for xi in Xinterp])

#Son 3, uno para el polinomio, datos originnales y los datos interpolados
plt.plot(Xint, yInt, "r", label="Newton")
plt.plot(xData, yData, "o", label="Datos")
plt.plot(Xinterp, yInterp, "ro", markersize=8, label="Puntos interpolados")
plt.xlabel("Velocidad (rpm)")
plt.ylabel("Amplitud (mm)")
plt.legend()
plt.grid()
plt.show()

print("  Xinterp   yInterp ")
print("------------------------------------------")
for i in range(len(Xinterp)):
    y = evalPoly(coeff, xData, Xinterp[i])
    print(" %.1f  %.8f" % (Xinterp[i], y))

    
    
    
    
    
    
    