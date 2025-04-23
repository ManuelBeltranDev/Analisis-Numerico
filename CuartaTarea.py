import numpy as np
import matplotlib.pyplot as plt
import math
from sympy import symbols, sin, cos, diff

### Ejercicio 1
"""
Escribe un programa que calcule todas las raíces 
de \( f(x) = 0 \) en un intervalo dado utilizando 
el **método de Ridder**. Usa las funciones `rootsearch` y `ridder`. 
Puedes usar el programa del **Ejemplo 4.3** como modelo.

Prueba el programa encontrando las raíces de:


f(x) = x sin(x) + 3 cos(x) - x


en el intervalo \( (-6, 6) \).

"""
#para conocer los rangos a y b de la raiz primero graficare

# Definir la función
def f(x):
    return x * math.sin(x) - 3 * math.cos(x) - x

# Crear un rango de valores para x, limitado entre -1 y 1
x_vals = np.linspace(-8, 8, 10000)

# Calcular los valores de y para la función f(x)
y_vals = [f(x) for x in x_vals]

# Graficar la función
plt.plot(x_vals, y_vals, label=r'$f(x) = x \cdot \sin(x) - 3 \cdot \cos(x) - x$')

# Añadir título y etiquetas
plt.title('Gráfico de la función f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')

# Mostrar cuadrícula y leyenda
plt.grid(True)
plt.legend()

# Mostrar el gráfico
plt.show()


"""No mire que desde un principio el problema ya establecia un rango 
pero graficando tambien te das cuenta de la o las raices de la funcion"""

## Modulo Newton-Raphson
## raiz = newtonRaphson(f,df,a,b,tol=1.0e-9).
## Encuentra la raiz de f(x) = 0 combinando Newton-Raphson
## con biseccion. La raiz debe estar en el intervalo (a,b).
## Los usuarios definen f(x) y su derivada df(x).
def err(string):
  print(string)
  input('Press return to exit')
  sys.exit()

def newtonRaphson(f,df,a,b,tol=1.0e-9):
  from numpy import sign
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa) == sign(fb): err('La raiz no esta en el intervalo')
  x = 0.5*(a + b)
  for i in range(30):
    print(i)
    fx = f(x)
    if fx == 0.0: return x 
    if sign(fa) != sign(fx): b = x # Haz el intervalo mas pequeño
    else: a = x
    dfx = df(x)  
    try: dx = -fx/dfx # Trata un paso con la expresion de Delta x
    except ZeroDivisionError: dx = b - a # Si division diverge, intervalo afuera
    x = x + dx # avanza en x
    if (b - x)*(x - a) < 0.0: # Si el resultado esta fuera, usa biseccion
      dx = 0.5*(b - a)
      x = a + dx 
    if abs(dx) < tol*max(abs(b),1.0): return x # Checa la convergencia y sal
  print('Too many iterations in Newton-Raphson')
print("Ejercicio 1 ---------------------------------------------------------------")

def f(x): return x*sin(x) - 3*cos(x) - x
def df(x): return 4*sin(x) + x*cos(x) - 1 
root = newtonRaphson( f, df, -2, 0)

print('Raiz =',root)


#Para comprobar que la dervada era la correcta  por que se hizo a mano 


x = symbols('x')
f = x*sin(x) - 3*cos(x) - x
derivada = diff(f, x)
print("La derivada es:", derivada)

#%%%%%%%%
print("Ejercicio 2 ---------------------------------------------------------------")
"""
La velocidad v de un cohete Saturn V en vuelo vertical, 
cerca de la superficie de la Tierra, 
se puede aproximar con la siguiente fórmula:

v(t) = u * ln(M0 / (M0 - m_dot * t)) - g * t

donde:
u = 2510 m/s        -> velocidad de escape de los gases respecto al cohete.
M0 = 2.8e6 kg       -> masa del cohete al momento del lanzamiento.
m_dot = 13.3e3 kg/s -> tasa de consumo de combustible.
g = 9.81 m/s^2      -> aceleración gravitacional.
t = tiempo transcurrido desde el despegue.

Pregunta:
Determinar el tiempo t cuando la velocidad v(t) 
alcanza la velocidad del sonido (335 m/s).
"""
#Como lo que tengo quue buscar es en que valor de t 
#la ecuacion es igual a cero (la raiz), primero restoa 
#ambos lados el valor de la velocidad 335 y la funcion
#quedara igualada a cero y lista


u = 2510
M0 = 2.8e6
m_dot = 13.3e3
g = 9.81
V = 335

#para ajustar el rango para encontrar la raiz de la funcion me ayudo
#graficando

def f(t):
    return u * np.log(M0 / (M0 - m_dot * t)) - g * t - V

# Crear un rango de valores para x, limitado entre -1 y 1
x_vals = np.linspace(65, 75, 10000)

# Calcular los valores de y para la función f(x)
y_vals = [f(x) for x in x_vals]

# Graficar la función
plt.plot(x_vals, y_vals, label=r'$u * np.log(M0 / (M0 - m_dot * t)) - g * t - V$')

# Añadir título y etiquetas
plt.title('Gráfico de la función f(x)')
plt.xlabel('x')
plt.ylabel('f(x)')

# Mostrar cuadrícula y leyenda
plt.grid(True)
plt.legend()

# Mostrar el gráfico
plt.show()


from numpy import sign
import sys

def ridder(f,a,b,tol=1.0e-9):
  fa = f(a)
  if fa == 0.0: return a
  fb = f(b)
  if fb == 0.0: return b
  if sign(fa) == sign(fb):
    print("No hay raíz en el intervalo — la función no cambia de signo.")
    return None
  xOld = None
  for i in range(30):
      c = 0.5*(a + b); 
      fc = f(c)
      s = math.sqrt(fc**2 - fa*fb)
      if s == 0.0: return None
      dx = (c - a)*fc/s
      if (fa - fb) < 0.0: dx = -dx
      x = c + dx; fx = f(x)
      if xOld is not None:
         if abs(x - xOld) < tol*max(abs(x),1.0): return x
      xOld = x
      if sign(fc) == sign(fx):
        if sign(fa)!= sign(fx): b = x; fb = fx
        else: a = x; fa = fx
      else:
        a = c; b = x; fa = fc; fb = fx
  print('Too many iterations')
  return None

def f(t): 
    return u * np.log(M0 / (M0 - m_dot * t)) - g * t - V


a = 70
b = 72

raiz = ridder(f, a, b)
print('raiz =',raiz)
#%%%%%%%
print("Ejercicio 3 ---------------------------------------------------------------")
"""
9. Usa los datos de la tabla para calcular f'(0.2) de la manera más 
precisa posible:

x     | 0       | 0.1     | 0.2     | 0.3     | 0.4
f(x)  | 0.00000 | 0.07835 | 0.13891 | 0.19292 | 0.24498
"""
#Segunda derivada de f con aproximación central
# Valores de la tabla
x = [0.0, 0.1, 0.2, 0.3, 0.4]
fx = [0.00000, 0.07835, 0.13891, 0.19292, 0.24498]

# Queremos f'(0.2) usando la aproximación más precisa posible
# Usamos la derivada centrada:
# f'(0.2) ≈ (f(x) - f(0.1)) / (2h)

h = 0.1
fp_02 = (fx[3] - fx[1]) / (2 * h)

print("f'(0.2) ≈ %.5f" % fp_02)





#%%%%%%%
print("Ejercicio 4 ---------------------------------------------------------------")
"""
10. Usando cinco cifras significativas en los cálculos, 
determina d(sin(x))/dx en x = 0.8 mediante:
(a) la primera aproximación por diferencia progresiva y
(b) la primera aproximación por diferencia central.
En cada caso, usa el valor de h que dé el
resultado más preciso (esto requiere experimentación).
"""
#Primero hay que saber cuanto es la derivada de seno evaluada en 0.8 
from sympy import symbols, diff, sin, lambdify
x = symbols('x')
f = sin(x)
fp = diff(f, x)
f = lambdify(x, fp)
print("valor correcto de la derivada de seno evaluada en x = 0.8, es: ", round(f(0.8),5))
# despues de eso a prueba y error encuentro la h de cada
 
# aproximacion con la que se acerque mas a el valor correcto
#(a) la primera aproximación por diferencia progresiva (forward)
def f(x,n): #La función a derivar con n decimales
  return round(sin(x),5)

def dff(x,h,f,n):
  dff= (-3*f(x, 5)+4*f(x + h,  5) - f(x + 2*h, 5))/ (2 * h)#Este salio de las tablas para derivadas más altas para diferencias finitas forward
  return dff

print(round(dff(0.8, 0.0380101,  f, 5), 5))

#(b) la primera aproximación por diferencia central 
def f(x,n): #La función a derivar con n decimales
  return round(sin(x),5)

def dff(x,h,f,n):
  dff= (f(x + h, 5) - f(x - h, 5)) / (2 * h)#Este salio de las tablas para los coeficientes de la aproximación central de diferencias finitas 
  return dff

print(round(dff(0.8, 0.02000102, f, 5), 5))

#despues de bastantes intentos esas h fueron con las que consegui que se acercara mas

#%%%%%%%
print("Ejercicio 5 ---------------------------------------------------------------")
"""Use the recursive trapezoidal rule to evaluate 3 π/4
0 ln(1 + tan x)dx. Explain the
results."""

#con el codigo

'''
Modulo regla trapezoidal recursiva

Inew = trapecio_recursiva(f,a,b,Iold,k).
Iold = Integral de f(x) de x = a hasta b calculada
con la regla trapezoidal recursiva con 2ˆ(k-1) paneles.
Inew = la misma integral calculada con 2ˆk paneles.
'''
def trapecio_recursiva(f,a,b,Iold,k):
  if k == 1: Inew = (f(a) + f(b))*(b - a)/2.0
  else:
    n = 2**(k -2 ) # numero de nuevos puntos
    h = (b - a)/n # espaciamiento de nuevos puntos
    x = a + h/2.0
    sum = 0.0
    for i in range(n):
      sum = sum + f(x)
      x = x + h
      Inew = (Iold + h*sum)/2.0
  return Inew

def f(x): return np.log(1 + np.tan(x))
Iold = 0.0
for k in range(1,21):
  Inew = trapecio_recursiva(f,0.0,np.pi/4,Iold,k)
  if (k > 1) and (abs(Inew - Iold)) < 1.0e-6: break
  Iold = Inew

print('Integral =',Inew)
print('n Panels =',2**(k-1))
print("Explicacion del resultado :")
print("El numero de panneles nos dice el numero por el cual se dividio a el area para facilitar su calculo y aumentar su precisión, entre mas paneles tengas mayor será la precisión, no se como se dividio en este caso pero pudo haber sido para el primero panel de 0 pi/8 y para el segundo de pi/8 hasta pi/4  ")
