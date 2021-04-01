# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Guillermo García Arredondo
"""

import numpy as np
import matplotlib.pyplot as plt

seed = 1
np.random.seed(seed)

print("EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS\n")
print("-Ejercicio 1-\n")

def E(u,v):
    return (u**3*np.exp(v-2)-2*v**2*np.exp(-u))**2   

# Derivada parcial de E con respecto a u
def dEu(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(3*u**2*np.exp(v-2)+2*v**2*np.exp(-u))
    
# Derivada parcial de E con respecto a v
def dEv(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(u**3*np.exp(v-2)-4*v*np.exp(-u))

# Gradiente de E (Ejercicio 1.2.a)
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# Ejercicio 1.1 - Implementar el algoritmo de gradiente descendente
def gradient_descent(initial_point, eta, error2get, maxIter, E, gradE):
    iterations = 0
    w = initial_point
    error = E(w[0], w[1])
    descenso = np.array([iterations, error],dtype=np.float64)
    while not abs(error) < error2get and iterations < maxIter:
        w = w - eta*gradE(w[0], w[1])
        error = E(w[0], w[1])
        iterations += 1
        descenso = np.row_stack((descenso, np.array([iterations, error])))
    return w, iterations, descenso 


# Ejercicio 1.2
eta = 0.1 
maxIter = 10000000000
error2get = 1e-14
initial_point = np.array([1.0,1.0],dtype=np.float64)
w, it, descenso = gradient_descent(initial_point, eta, error2get, maxIter, E, gradE)

# 1.2.b - ¿Cuántas iteraciones tarda el algoritmo en obtener por primera vez un
# valor de E(u,v) inferior a 10^-14?
print ("-1.2-\n\nNúmero de iteraciones: ", it)
# 1.2.c - ¿En qué coordenadas (u,v) se alcanzó por primera vez un valor igual o
# menor a 10^-14 en el apartado anterior?
print ("Coordenadas obtenidas: (", w[0], ", ", w[1], ")")

# DISPLAY FIGURE
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
Z = E(X, Y) #E_w([X, Y])
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor="none", rstride=1,
                        cstride=1, cmap="jet")
min_point = np.array([w[0],w[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], E(min_point_[0], min_point_[1]), "r*", markersize=10)
ax.set(title="Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente")
ax.set_xlabel("u")
ax.set_ylabel("v")
ax.set_zlabel("E(u,v)")
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 1.3.a ---\n")

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
# como máximo. Repetir para eta = 0.1 .
maxIter = 50
initial_point = np.array([-1.0, 1.0], dtype=np.float64)
w1, it1, descenso1 = gradient_descent(initial_point, 0.01, error2get, maxIter, F, gradF)
w2, it2, descenso2 = gradient_descent(initial_point, 0.1, error2get, maxIter, F, gradF)

print ("Para eta = 0.01\nNúmero de iteraciones: ", it1)
print ("Coordenadas obtenidas: (", w1[0], ", ", w1[1], ")")
maxIter = 1000
print ("\nPara eta = 0.1\nNúmero de iteraciones: ", it2)
print ("Coordenadas obtenidas: (", w2[0], ", ", w2[1], ")")

plt.plot(descenso1[:,0], descenso1[:,1])
plt.plot(descenso2[:,0], descenso2[:,1], "r")
plt.title("Ejercicio 1.3.a. Gradiente descendente de F(x,y)")
plt.xlabel("Iteraciones")
plt.ylabel("F(x,y)")
plt.legend(("eta = 0.01", "eta = 0.1"))
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 1.3.b ---\n")
print("Se muestra tabla...")

eta = 0.01
a, it, des = gradient_descent(np.array([-0.5, -0.5],dtype=np.float64), eta, error2get, maxIter, F, gradF)
b, it, des = gradient_descent(np.array([1.0, 1.0],dtype=np.float64), eta, error2get, maxIter, F, gradF)
c, it, des = gradient_descent(np.array([2.1, -2.1],dtype=np.float64), eta, error2get, maxIter, F, gradF)
d, it, des = gradient_descent(np.array([-3.0, 3.0],dtype=np.float64), eta, error2get, maxIter, F, gradF)
e, it, des = gradient_descent(np.array([-2.0, 2.0],dtype=np.float64), eta, error2get, maxIter, F, gradF)
valores = np.array([np.append(a, F(a[0], a[1])), np.append(b, F(b[0], b[1])),
                    np.append(c, F(c[0], c[1])), np.append(d, F(d[0], d[1])),
                    np.append(e, F(e[0], e[1]))])

fig, ax = plt.subplots()
ax.axis("off")
table = ax.table(cellText=valores, 
          colLabels=["x","y","F(x,y)"],
          rowLabels=["(-0.5,-0.5)",
                     "(1,1)",
                     "(2.1,-2.1)",
                     "(-3,3)",
                     "(-2,2)"],
          loc="center")

plt.title("Ejercicio 1.3.b. Para eta = " +str(eta)+ " y un máximo de " +str(maxIter)+ " iteraciones")
table.scale(2.5,2.5)
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2 ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print("EJERCICIO SOBRE REGRESIÓN LINEAL\n")
print("-Ejercicio 2-\n")
from sklearn.utils import shuffle

label5 = 1
label1 = -1

# Función para leer los datos
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

# Función para calcular el error
def Err(x,y,w):
    return ((np.matmul(x,w) - y)**2).mean(axis=0)

# Gradiente de la función de error
def gradErr(x, y, w):
    return (2/len(x)*np.matmul(x.T, (np.matmul(x,w) - y)))
    
# Gradiente Descendente Estocástico
def sgd(initial_point, x, y, eta, error2get, maxIter, minibatch_size):
    w = initial_point  
    iterations = 0
    
    parar = False
    while not parar and iterations < maxIter:
        x,y = shuffle(x, y, random_state=seed)
        minibatches_x = np.array_split(x, len(x)//minibatch_size)
        minibatches_y = np.array_split(y, len(y)//minibatch_size)
        
        for i in range(0, len(minibatches_x)):
            w = w - eta*gradErr(minibatches_x[i], minibatches_y[i], w)
            error = Err(minibatches_x[i], minibatches_y[i], w)
            parar = abs(error) < error2get
            iterations += 1
            if parar:
                break
    return w

# Pseudoinversa	
def pseudoinverse(x, y):
    w = np.matmul(np.linalg.pinv(x), y)
    return w

# Ejercicio 2.1 - Estimar un modelo de regresión lineal usando tanto el
# algoritmo de gradiente descendente estócastico como la pseudo-inversa.
print("-2.1-\n")

# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

maxIter = 1000
w_sgd = sgd(np.array([0.0] * x.shape[1], dtype=np.float64), x, y, eta, error2get, maxIter, 24)
print ("Bondad del resultado para grad. descendente estocástico:")
print ("Ein: ", Err(x,y,w_sgd))
print ("Eout: ", Err(x_test, y_test, w_sgd))

w_pinv = pseudoinverse(x, y)
print ("\nBondad del resultado para la pseudoinversa:")
print ("Ein: ", Err(x,y,w_pinv))
print ("Eout: ", Err(x_test, y_test, w_pinv))

sgd_x = np.linspace(0, 1, 2)
sgd_y = (-w_sgd[0]-w_sgd[1]*sgd_x)/w_sgd[2]

pinv_x = np.linspace(0, 1, 2)
pinv_y = (-w_pinv[0]-w_pinv[1]*pinv_x)/w_pinv[2]

plt.plot(sgd_x, sgd_y, c="blue")
plt.plot(pinv_x, pinv_y, c="red")
plt.scatter(x[np.where(y == label5), 1], x[np.where(y == label5), 2], c="yellow")
plt.scatter(x[np.where(y == label1), 1], x[np.where(y == label1), 2], c="purple")
plt.legend(("SGD", "Pinv", "5", "1"))
plt.title("Ejercicio 2.1. SGD vs. Pseudoinversa para X de entrenamiento")
plt.xlabel("Intensidad promedio")
plt.ylabel("Simetría")
plt.show()

input("\n--- Pulsar tecla para mostrar la gráfica con la muestra de prueba ---\n")
print("Se muestra gráfica...")

plt.plot(sgd_x, sgd_y, c="blue")
plt.plot(pinv_x, pinv_y, c="red")
plt.scatter(x_test[np.where(y_test == label5), 1], x_test[np.where(y_test == label5), 2], c="yellow")
plt.scatter(x_test[np.where(y_test == label1), 1], x_test[np.where(y_test == label1), 2], c="purple")
plt.legend(("SGD", "Pinv", "5", "1"))
plt.title("Ejercicio 2.1. SGD vs. Pseudoinversa para X de prueba")
plt.xlabel("Intensidad promedio")
plt.ylabel("Simetría")
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2.a ---\n")

# Ejercicio 2.2

# 2.2.a - Función que muestrea datos uniformemente en un cuadrado [-size,size]x[-size,size]

def simula_unif(N, d, size):
 	return np.random.uniform(-size,size,(N,d))
 
x_train = simula_unif(1000, 2, 1)
print("La muestra de entrenamiento: ")
print(x_train)
plt.scatter(x_train[:,0], x_train[:,1])
plt.title("Ejercicio 2.2.a. Muestra de entrenamiento uniforme")
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2.b---\n")

# 2.2.b. - Asignar etiquetas dado f(x1, x2) e introducir ruido sobre 10% de las
# mismas
print("Se muestra gráfica...")

def sign(x):
 	if x >= 0:
         return 1
 	return -1

def f(x1, x2):
 	return sign((x1-0.2)**2 + x2**2 - 0.6)

def generar_ruido(y):
    rng = np.random.default_rng()
    for i in range(0, y.size//10):
        y[rng.choice(y.size, replace=False)] *= -1

y_train = np.array([f(x1, x2) for x1, x2 in x_train], dtype=np.float64)
plt.scatter(x_train[np.where(y_train == 1), 0], x_train[np.where(y_train == 1), 1], c="yellow")
plt.scatter(x_train[np.where(y_train == -1), 0], x_train[np.where(y_train == -1), 1], c="purple")
plt.legend(("+1","-1"), loc="upper right")
plt.title("Ejercicio 2.2.b - Muestra etiquetada sin ruido")
plt.show()

input("\n--- Pulsar tecla para continuar con el ejercicio 2.2.b---\n")
print("Se muestra gráfica...")

generar_ruido(y_train)
plt.scatter(x_train[np.where(y_train == 1), 0], x_train[np.where(y_train == 1), 1], c="yellow")
plt.scatter(x_train[np.where(y_train == -1), 0], x_train[np.where(y_train == -1), 1], c="purple")
plt.legend(("+1","-1"), loc="upper right")
plt.title("Ejercicio 2.2.b - Muestra etiquetada con ruido")
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2.c---\n")

# 2.2.c - Ajustar un modelo de regresión lineal al conjunto de datos generado y estimar w con SGD

x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
w_sgd = sgd(np.array([0.0] * x_train.shape[1], dtype=np.float64), x_train, y_train, eta, error2get, maxIter, 24)
print("Ein: ", Err(x_train, y_train, w_sgd))

sgd_x = np.linspace(-1, 1, 2)
sgd_y = (-w_sgd[0]-w_sgd[1]*sgd_x)/w_sgd[2]

plt.plot(sgd_x, sgd_y, c="red")
plt.ylim(-1.,1.)
plt.scatter(x_train[np.where(y_train == 1), 1], x_train[np.where(y_train == 1), 2], c="yellow")
plt.scatter(x_train[np.where(y_train == -1), 1], x_train[np.where(y_train == -1), 2], c="purple")
plt.legend(("SGD", "+1", "-1"), loc="upper right")
plt.title("Ejercicio 2.2.c Regresión lineal para la muestra generada")
plt.show()