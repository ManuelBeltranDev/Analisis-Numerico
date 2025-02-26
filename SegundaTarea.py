import numpy as np
"""
Intersección de trayectorias.
Tres objetos se mueven de tal manera que sus trayectorias son:
    
2x-y+3z =24
2y-z=14
7x-5y=6
    
Encontrar su punto de intersección."""
a = np.array([[2,-1,3],[0,2,-1],[7,-5,0]])
b = np.array([[24],[14],[6]])

def gaussElimin(a,b):
  n = len(b)
  # Fase de eliminacion
  for k in range(0,n-1):
    for i in range(k+1,n):
      if a[i,k] != 0.0:
        lam = a [i,k]/a[k,k]
        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
        b[i] = b[i] - lam*b[k]
  # Fase de sustitucion hacia atras
  for k in range(n-1,-1,-1):
    b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
  return b

print(gaussElimin(a,b))


#%%

"""Carga de los quarks
Los protones y neutrones están formados cada uno por tres quarks. 
Los protones poseen dos quarks up (u) y un quark down (d), 
los neutrones poseen un quark up y dos quarks down. 
Si la carga de un protón es igual al positivo de la carga del electrón (+e) 
 y la carga de un neutrón es cero, determine 
 las cargas de los quarks up y down. (Tip: suponga que +e = 1.)"""
 
p = np.array([[2,1],[1,2]])
n = np.array([[1],[0]])
p = p.astype(float)
n = n.astype(float)

def gaussElimin(a,b):
  n = len(b)
  # Fase de eliminacion
  for k in range(0,n-1):
    for i in range(k+1,n):
      if a[i,k] != 0.0:
        lam = a [i,k]/a[k,k]
        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
        b[i] = b[i] - lam*b[k]
  # Fase de sustitucion hacia atras
  for k in range(n-1,-1,-1):
    b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
  return b

print(gaussElimin(a = p,b = n))
#%%
"""
Meteoros
El Centro de Investigación 1 examina la cantidad de meteoros 
que entran a la atmósfera. Con su equipo de recopilación de datos 
durante 8 horas captó 95kg de meteoros, por fuentes externas sabemos 
que fueron de 4 distintas masas (1kg, 5kg, 10kg y 20kg). La cantidad 
total de meteoros fue de 26. Otro centro de investigación captó que la 
cantidad de meteoros de 5kg es 4 veces la cantidad de meteoros de 10kg,
 y el número de meteoros de 1kg es 1 menos que el doble de la cantidad
 de meteoros de 5kg. Después use matrices para encontrar el número asociado
 a cada masa de meteoros.
"""


F = np.array([[ 1, 5, 10, 20],[ 0, 1,-4, 0],[ -1, 2, 0, 0],[ 1, 1, 1, 1]])
G = np.array([[95],[0],[1],[26]])
print(gaussElimin( F, G))






























