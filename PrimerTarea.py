import math


"Ejercicio 1"

"""Elabora un programa que calcule la temperatura en grados 
Celsius a partir de temperaturas conocidas (dadas) en grados Farenheit."""


def FaCaF():
    c = 2
    X=input("Para la conversion de Fahrenheit a Celsius ingrese (FaC), Para la conversion Celsius de  a Fahrenheit ingrese (CaF): ")
    while c !=1:
        if X== "FaC":
            F = float(input("Ingresa la cantidad de grados Farhenheit: "))
            C = (F-32)/1.8
            C= round(C, 3)
            print("Tu conversion de" ,F , " Farhenheit a  Celsius es ", C ) 
            c = 1
        elif X== "CaF":
            C = int(input("Ingresa la cantidad de grados Celsius: "))
            F = (C*1.8)+32
            F = round(F, 3)
            print("Tu conversion de" ,C , " Celsius  a  Farhenheit es ", F)
            c = 1
        else:
            print("Incorrecto, ingrese una de las opciones:")
            X = input("Para la conversion de Fahrenheit a Celsius ingrese (FaC), Para la conversion Celsius de  a Fahrenheit ingrese (CaF): ")
#FaCaF()

#%%
"Ejercicio 2"

"""
Considera la definición del sinh(x)

            sinh(x) = (e^x - e^(-x)) / 2

Elabora un programa que calcule el valor de sinh(x) en x = 2π de tres diferentes maneras:
"""

"1. Evaluando sinh(x) directamente."

def seno():
    x = 5
    V = math.sinh(x)
    print("El seno hiperbolico evaluado en 5 es:" ,V)
    
    
    "2. Evaluando con la definición del lado derecho, usando la función exponencial."
    
    
    A = ((math.exp(x))-math.exp(-x))/2
    print("La ecuacion evaluada en 5 es igual a", A)
    
    
    "3. Evaluando con la definición del lado derecho, usando la potencia."
    
    
    B = ((math.e**(x))-math.e**(-x))/2
    print("La ecuacion evaluada en 5 es igual a", B)
#seno()

#%%

"""Ejercicio 3"""




"""1. Considera la relación entre el seno en variable compleja y el seno hiperbólico en variable real x,

            sin(ix) = i sinh(x).

Elabora un programa que calcule el valor de sin(ix) y de sinh(x) para ciertos valores dados de x, para verificar la identidad."""

from cmath import sin, sinh, cos
def ver():
    x = 5
    ix = (1j)*(x)
    sen = sin(ix)
    print("El resultado de seno evaluado en el numero complejo", ix, "es", sen)
    
    seh = (1j)*(sinh(x))
    print("El resultado de el producto entre el numero complejo",1j , "el seno hiperbolico evaluado en", x, "es", seh, "El resultado es identico por lo tanto la identidad se cumple.")
    
    
    """
    2. Considera la relación de Euler para x real,
    
                e^(ix) = cos(x) + i sin(x).
    
    Elabora un programa que calcule el valor de cos(x), sin(x) y de e^(ix) para ciertos valores dados de x, para verificar la identidad.
    """
    
    e = math.e**(ix)
    print("El resultado de el exponencial evaluado en el numero", ix,"es:", e)
    
    E = cos(x) + (1j)*(sin(x))
    print("El resultado de cos(x) + i sin(x) es:", E, "   Identidad verificada")
#ver()
#%%
import numpy as np
def general():
    """Ejercicio 4
    
    Este tratamiento flexible de funciones en el plano complejo permite encontrar las raíces reales o complejas de una función cuadrática.
    
    Considera que las raíces de f(z) = az² + bz + c se obtienen
    
               z± = (-b ± √(b² - 4ac)) / (2a).
    
    Elabora un programa en el que uses Numpy para calcular el valor de las raíces con diferentes valores dados de a, b, c, para obtener ejemplos de raíces reales y complejas."""
    
    a , b , c = float(input("Ingresa el coeficiente de el termino cuadratico: ")), float(input("Ingresa el coeficiente de el termino lineal: ")), float(input("Ingresa el termino constante: "))
    Discriminante = b**2 - 4*a*c
    
    raiz1 = (-b + np.lib.scimath.sqrt(Discriminante))/(2*a)
    raiz2 = (-b - np.lib.scimath.sqrt(Discriminante))/(2*a)
    print(f"Las raices de la ecuación {a}x^2 + {b}x + {c} = 0, son {raiz2} y {raiz1}" )
#general()
#%%

def tray():
    """Ejercicio 5
    
    ¿Cuál es la trayectoria de una pelota que se lanza con una rapidez inicial v0 y un ángulo θ medido de la horizontal?
    
    Sabemos que la pelota seguirá una trayectoria y = f(x), donde, en ausencia de resistencia del aire,
    
        f(x) = x * tan(θ) - (g / (2 * v0**2 * cos²(θ))) * x² + y0.
    
    En esta expresión, x es la coordenada horizontal, g es la aceleración de la gravedad y y0 es la posición inicial de la pelota.
    
    1. En tu portafolio de clase, elabora un programa en el que evalúes esta expresión. 
       El programa debe escribir el valor de todas las variables involucradas junto con las unidades usadas.
    """
    
    x = 5  
    θ = math.radians(45) 
    g = 9.81
    v0 = 50  
    y0 = 124
    
    
    y = x * math.tan(θ) - (g / (2 * v0**2 * math.cos(θ)**2)) * x**2 + y0
    
    print(f"Valor en coordenada horizontal x = {x} m")
    print(f"Valor de el angulo en radianes {θ} rad")
    print(f"Valor de la aceleracion de la gravedad en la tierra g = {g} m/s^2")
    print(f"Valor de la velocidad inicial de la pelota v0 = {v0} m/s")
    print(f"Valor de la altura inicial y0 = {y0} m")
    print(f"Valor de la trayectoria en y para x = {x} m: {y} m")

#tray()





























