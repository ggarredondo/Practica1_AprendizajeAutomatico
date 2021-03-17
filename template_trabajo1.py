# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Guillermo García Arredondo
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

print('EJERCICIO SOBRE LA BUSQUEDA ITERATIVA DE OPTIMOS\n')
print('Ejercicio 1\n')

def E(u,v):
    return (u**3*np.exp(v-2)-2*v**2*np.exp(-u))**2   

#Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(3*u**2*np.exp(v-2)+2*v**2*np.exp(-u))
    
#Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(u**3*np.exp(v-2)-4*v*np.exp(-u))

#Gradiente de E (Ejercicio 1.2.a)
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# Ejercicio 1.1 - Implementar el algoritmo de gradiente descendente
def gradient_descent(initial_point, eta, error2get, maxIter, E, gradE):
    iterations = 0
    w = initial_point
    error = E(w[0], w[1])
    descenso_p = [w]
    descenso_E = [error]
    while not error < error2get and iterations < maxIter:
        w = w - eta*gradE(w[0], w[1])
        error = E(w[0], w[1])
        descenso_p.append(w)
        descenso_E.append(error)
        iterations += 1
    return w, iterations, descenso_p, descenso_E    


# Ejercicio 1.2
eta = 0.1 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0])
w, it, descenso_p, descenso_E = gradient_descent(initial_point, eta, error2get, maxIter, E, gradE)

# 1.2.b - ¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un
# valor de E(u,v) inferior a 10^-14?
print ('Número de iteraciones: ', it)
# 1.2.c - ¿En qué coordenadas (u,v) se alcanzó por primera vez un valor igual o
# menor a 10^-14 en el apartado anterior?
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')


# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Ejercicio 1.3

def F(x, y):
    return (x+2)**2 + 2*(y-2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) 

def dFx(x, y):
    return 2*(x+2) + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)

def dFy(x, y):
    return 4*(y-2) + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

def gradF(x, y):
    return np.array([dFx(x, y), dFy(x, y)])

# 1.3.a - Minimizar la función para (x0 = -1, y0 = 1), eta = 0.01 y 50 iteraciones
# como máximo
eta = 0.01
maxIter = 50
initial_point = np.array([-1.0, 1.0])
w, it, descenso_p, descenso_E = gradient_descent(initial_point, eta, error2get, maxIter, F, gradF)


print ('Numero de iteraciones: ', it)
print ('Coordenadas obtenidas: (', w[0], ', ', w[1],')')



###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('EJERCICIO SOBRE REGRESION LINEAL\n')
print('Ejercicio 1\n')

label5 = 1
label1 = -1

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

# Funcion para calcular el error
def Err(x,y,w):
    return 

# Gradiente Descendente Estocastico
def sgd(?):
    #
    return w

# Pseudoinversa	
def pseudoinverse(?):
    #
    return w


# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')


w = sgd(?)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

input("\n--- Pulsar tecla para continuar ---\n")

#Seguir haciendo el ejercicio...

print('Ejercicio 2\n')
# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))

def sign(x):
	if x >= 0:
		return 1
	return -1

def f(x1, x2):
	return sign(?) 

#Seguir haciendo el ejercicio...



