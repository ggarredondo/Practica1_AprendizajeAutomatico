# -*- coding: utf-8 -*-
"""
TRABAJO 1. 
Nombre Estudiante: Guillermo García Arredondo
"""

import numpy as np # Biblioteca numpy que nos permite trabajar con vectores y funciones trigonométricas.
import matplotlib.pyplot as plt # Pyplot para hacer las gráficas/tablas.

# Establezco la semilla para los números pseudoaleatorios de numpy para que los resultados obtenidos
# sean reproducibles. Asigno la semilla a una variable 'seed' porque luego la tendremos que introducir
# de nuevo.
seed = 1
np.random.seed(seed)

print("EJERCICIO SOBRE LA BÚSQUEDA ITERATIVA DE ÓPTIMOS\n")
print("-Ejercicio 1-\n")

# Función E del ejercicio 1.2.
def E(u,v):
    return (u**3*np.exp(v-2)-2*v**2*np.exp(-u))**2   

# Derivada parcial de E con respecto a u.
def dEu(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(3*u**2*np.exp(v-2)+2*v**2*np.exp(-u))
    
# Derivada parcial de E con respecto a v.
def dEv(u,v):
    return 2*(u**3*np.exp(v-2)-2*v**2*np.exp(-u))*(u**3*np.exp(v-2)-4*v*np.exp(-u))

# Gradiente de E (Ejercicio 1.2.a).
def gradE(u,v):
    return np.array([dEu(u,v), dEv(u,v)])

# Ejercicio 1.1 - Implementar el algoritmo de gradiente descendente.
# Como argumentos de entrada tenemos el punto inicial desde donde el gradiente descendente
# va a comenzar a minimizar, la tasa de aprendizaje η (eta), el valor mínimo buscado,
# el número de iteraciones máxima, la función a minimizar y el gradiente de la función.
# La función y el gradiente son también argumentos para poder reutilizar el código
# en el apartado 1.3.
def gradient_descent(initial_point, eta, error2get, maxIter, E, gradE):
    iterations = 0 # Inicializamos el número de iteraciones a 0.
    w = initial_point # Asignamos a w el punto inicial.
    error = E(w[0], w[1]) # Calculamos el valor de la función dado el punto inicial.
    descenso = np.array([iterations, error],dtype=np.float64) # Guardamos en un vector el número de iteraciones y
                                                              # el valor de la función actual para su posterior
                                                              # visualización.
                                                              
    # Mientras el valor actual de la función no sea menor que el valor mínimo buscado y el número de
    # iteraciones no superen las iteraciones máximas. Se hace valor absoluto a 'error' para el caso
    # en el que la función dada como E pueda dar valores negativos, con tal de que no termine prematuramente
    # en un mínimo local.                                                       
    while not abs(error) < error2get and iterations < maxIter:
        w = w - eta*gradE(w[0], w[1]) # Obtenemos el nuevo w dada la ecuación general wj = wj-η*dEin(w)/dwj .
        error = E(w[0], w[1]) # Calculamos el valor de la función dada la nueva w.
        iterations += 1 # Contamos una iteración más.
        descenso = np.row_stack((descenso, np.array([iterations, error]))) # Guardamos en un vector el número de iteraciones y
                                                                           # el valor de la función actual para su posterior
                                                                           # visualización.
    # Se devuelve el w obtenido junto al número de iteraciones final y el descenso seguido por el algoritmo.
    return w, iterations, descenso


# Ejercicio 1.2.
# Para este ejercicio establecemos una tasa de aprendizaje de 0.1, 10000000000 iteraciones
# máximas, un valor mínimo de 10^-14 y un punto inicial (1, 1).
# La variable 'error2get' se seguirá usando durante el resto de la práctica sin cambiar su valor.
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

# Mostrar en una gráfica el hiperplano y el punto mínimo obtenido
from mpl_toolkits.mplot3d import Axes3D # Importamos Axes3D para poder visualizar el hiperplano en una
                                        # gráfica en tres dimensiones.
                                        
x = np.linspace(-2, 2, 50) # Obtenemos 50 valores equidistantes en el intervalo [-2, 2] en vez de [-30, 30]
                           # para que la gráfica sea más clara
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

# Función F del ejercicio 1.3.
def F(x, y):
    return (x+2)**2 + 2*(y-2)**2 + 2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) 

# Derivada parcial de F con respecto a x
def dFx(x, y):
    return 2*(x+2) + 4*np.pi*np.cos(2*np.pi*x)*np.sin(2*np.pi*y)

# Derivada parcial de F con respecto a y
def dFy(x, y):
    return 4*(y-2) + 4*np.pi*np.sin(2*np.pi*x)*np.cos(2*np.pi*y)

# Gradiente de F
def gradF(x, y):
    return np.array([dFx(x, y), dFy(x, y)])

# 1.3.a - Minimizar la función para (x0 = -1, y0 = 1), η = 0.01 y 50 iteraciones
# como máximo. Repetir para η = 0.1 .

# Establecemos el máximo de iteraciones a 50 y calculamos el gradiente descendente empezando por
# (-1, 1) para una tasa de aprendizaje 0.01 y 0.1
maxIter = 50
initial_point = np.array([-1.0, 1.0], dtype=np.float64)
w1, it1, descenso1 = gradient_descent(initial_point, 0.01, error2get, maxIter, F, gradF)
w2, it2, descenso2 = gradient_descent(initial_point, 0.1, error2get, maxIter, F, gradF)

# Imprimimos por pantalla los resultados obtenidos para η = 0.01
print ("Para eta = 0.01\nNúmero de iteraciones: ", it1)
print ("Coordenadas obtenidas: (", w1[0], ", ", w1[1], ")")
print ("Ein: ", F(w1[0], w1[1]))

# Y para η = 0.1
print ("\nPara eta = 0.1\nNúmero de iteraciones: ", it2)
print ("Coordenadas obtenidas: (", w2[0], ", ", w2[1], ")")
print ("Ein: ", F(w2[0], w2[1]))

# Y mostramos ambos descensos en una gráfica, dados las iteraciones y los valores 
# obtenidos durante la ejecución del algoritmo.
plt.plot(descenso1[:,0], descenso1[:,1])
plt.plot(descenso2[:,0], descenso2[:,1], "r")
plt.title("Ejercicio 1.3.a. Gradiente descendente de F(x,y)")
plt.xlabel("Iteraciones")
plt.ylabel("F(x,y)")
plt.legend(("eta = 0.01", "eta = 0.1"))
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 1.3.b ---\n")
print("Se muestra tabla...")

# Cambiamos la tasa de aprendizaje a 0.01 y calculamos el descenso para todos los puntos pedidos.
eta = 0.01
a, it, des = gradient_descent(np.array([-0.5, -0.5],dtype=np.float64), eta, error2get, maxIter, F, gradF)
b, it, des = gradient_descent(np.array([1.0, 1.0],dtype=np.float64), eta, error2get, maxIter, F, gradF)
c, it, des = gradient_descent(np.array([2.1, -2.1],dtype=np.float64), eta, error2get, maxIter, F, gradF)
d, it, des = gradient_descent(np.array([-3.0, 3.0],dtype=np.float64), eta, error2get, maxIter, F, gradF)
e, it, des = gradient_descent(np.array([-2.0, 2.0],dtype=np.float64), eta, error2get, maxIter, F, gradF)

# Concatenamos los puntos obtenidos con sus respectivos valores de F en un mismo vector para las columnas
# de la tabla.
valores = np.array([np.append(a, F(a[0], a[1])), np.append(b, F(b[0], b[1])),
                    np.append(c, F(c[0], c[1])), np.append(d, F(d[0], d[1])),
                    np.append(e, F(e[0], e[1]))])
# Con plt.subplots() obtenemos ax que son los ejes de la figura que necesitamos para crear la tabla.
fig, ax = plt.subplots()
ax.axis("off") # Pyplot por defecto genera los ejes de un gráfico 2D al crear una figura. Es por ello que tenemos que
               # desactivarlos si no queremos que estén de fondo por debajo de la tabla.
# Finalmente, creamos y mostramos la tabla utilizando el vector de columnas que habíamos inicializado antes 
# y añadiendo las etiquetas pertinentes.
table = ax.table(cellText=valores, 
          colLabels=["x","y","F(x,y)"],
          rowLabels=["(-0.5,-0.5)",
                     "(1,1)",
                     "(2.1,-2.1)",
                     "(-3,3)",
                     "(-2,2)"],
          loc="center")
plt.title("Ejercicio 1.3.b. Para eta = " +str(eta)+ " y un máximo de " +str(maxIter)+ " iteraciones")
table.scale(2.5,2.5) # Escalamos la tabla porque por defecto es demasiado pequeña y no es legible.
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2 ---\n")

###############################################################################
###############################################################################
###############################################################################
###############################################################################
print("EJERCICIO SOBRE REGRESIÓN LINEAL\n")
print("-Ejercicio 2-\n")
from sklearn.utils import shuffle # Para este ejercicio he importado de sklearn la función shuffle
# para poder barajar dos vectores en el mismo orden, pues es necesario para el algoritmo de
# gradiente descendente estocásitco.

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

# Función para calcular el error. E(hw) = 1/N*Σ(hw(xn) - yn)^2
# Se utiliza matmul para la correcta multiplicación de matrices de 'x' y 'w', lo cual
# nos da una fila de los valores estimados a la cual restamos 'y'. Hacemos el cuadrado
# de la diferencia y hacemos la media.
def Err(x,y,w):
    return ((np.matmul(x,w) - y)**2).mean(axis=0)

# Gradiente de la función de error. dE(hw)/dwj = 2/N*Σ(xnj*(hw(xn) - yn))
# Se hace la traspuesta de X para poder multiplicar con el resultado de X*w-y.
def gradErr(x, y, w):
    return (2/len(x)*np.matmul(x.T, (np.matmul(x,w) - y)))
    
# Gradiente Descendente Estocástico (SGD).
# Como argumentos de entrada tenemos el punto de inicio para empezar a minimizar,
# los datos x e y, la tasa de aprendizaje η (eta), el error mínimo buscado,
# el número máximo de iteraciones y el tamaño que tendrán los minibatches.
def sgd(initial_point, x, y, eta, error2get, maxIter, minibatch_size):
    w = initial_point # Asignamos el punto inicial a w.
    iterations = 0 # Inicializamos el número de iteraciones a 0.
    
    parar = False # Inicializamos una condición de parada que será la que compruebe
                  # más adelante si se ha alcanzado el mínimo.
    while not parar and iterations < maxIter: # Mientras no 'parar' y el número de iteraciones no alcance el máximo...
        x,y = shuffle(x, y, random_state=seed) # Barajamos los datos x e y utilizando la función shuffle
                                               # para asegurarnos que no se pierda la correspondencia. Además,
                                               # asignamos la semilla que previamente habíamos establecido. Esto no
                                               # causará que en cada llamada se baraje de la misma manera, solo que
                                               # los resultados sean reproducibles. Es decir, la aleatoridad se 
                                               # repite por ejecución pero no por llamada.
        # Divido x e y en n minibatches, siendo n el tamaño de x/y dividido entre el tamaño del minibatch                
        minibatches_x = np.array_split(x, len(x)//minibatch_size)
        minibatches_y = np.array_split(y, len(y)//minibatch_size)
        
        for i in range(0, len(minibatches_x)): # Para cada minibatch...
            w = w - eta*gradErr(minibatches_x[i], minibatches_y[i], w) # Obtenemos el nuevo w dado wj = wj-η*dEin(w)/dwj .
            error = Err(minibatches_x[i], minibatches_y[i], w) # Obtenemos el error dado el nuevo w
            parar = abs(error) < error2get # Comprobamos si el error es menor que el error mínimo buscado.
            if parar: # Si lo es, terminamos el bucle.
                break
        iterations += 1 # Tras iterar para cada minibatch, contamos una iteración y empieza de nuevo el bucle while.
    return w # Devolvemos el w final.

# Pseudoinversa. w = X†*y
# Para ello simplemente utilizamos matmul para multiplicar la pseudoinversa dada por
# la función pinv de numpy con y.
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

# Utlizamos el gradiente descendente estócastico para los datos de entrenamiento con la
# misma tasa de aprendizaje anterior (0.01), error mínimo 10^-14, 100 iteraciones máximas
# y 24 elementos por minibatch. Escribimos el resultado en pantalla.
maxIter = 100
w_sgd = sgd(np.array([0.0] * x.shape[1], dtype=np.float64), x, y, eta, error2get, maxIter, 24)
print ("Bondad del resultado para grad. descendente estocástico:")
print ("Ein: ", Err(x,y,w_sgd))

# Utilizamos la pseudoinversa para los mismos datos de entrenamiento y
# escribimos el resultado en pantalla.
w_pinv = pseudoinverse(x, y)
print ("\nBondad del resultado para la pseudoinversa:")
print ("Ein: ", Err(x,y,w_pinv))

# Una vez obtenidos los w para SGD y la pseudoinversa, vamos a mostrar las respectivas rectas
# en una gráfica.
sgd_x = np.linspace(0, 1, 2) # Para ello obtendremos dos puntos equidistantes en el intervalo [0,1],
                             # para la 'x' en la gráfica.
sgd_y = (-w_sgd[0]-w_sgd[1]*sgd_x)/w_sgd[2] # Y despejaremos x2 del vector de características w0 + w1*x1 + w2*x2,
                                            # para la 'y' en la gráfica.

# Hacemos lo mismo para el w obtenido con la pseudoinversa.
pinv_x = np.linspace(0, 1, 2)
pinv_y = (-w_pinv[0]-w_pinv[1]*pinv_x)/w_pinv[2]

# Y mostramos en una gráfica los puntos con sus respectivas etiquetas y las rectas
# dadas por el SGD y la pseudoinversa.
plt.plot(sgd_x, sgd_y, c="blue")
plt.plot(pinv_x, pinv_y, c="red")
plt.scatter(x[np.where(y == label5), 1], x[np.where(y == label5), 2], c="yellow")
plt.scatter(x[np.where(y == label1), 1], x[np.where(y == label1), 2], c="purple")
plt.legend(("SGD", "Pinv", "5", "1"))
plt.title("Ejercicio 2.1. SGD vs. Pseudoinversa para X de entrenamiento")
plt.xlabel("Intensidad promedio")
plt.ylabel("Simetría")
plt.show()

# Ahora escribimos por pantalla los resultados obtenidos para los datos de prueba y
# dibujamos una nueva gráfica.
input("\n--- Pulsar tecla para mostrar la gráfica con la muestra de prueba ---\n")
print ("Bondad del resultado para grad. descendente estocástico:")
print ("Eout: ", Err(x_test, y_test, w_sgd))
print ("\nBondad del resultado para la pseudoinversa:")
print ("Eout: ", Err(x_test, y_test, w_pinv))

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

# 2.2.a - Función que muestrea datos uniformemente en un cuadrado [-size,size]x[-size,size] .

def simula_unif(N, d, size):
 	return np.random.uniform(-size,size,(N,d))

# Obtenemos una muestra de entrenamiento utilizando simula_inf, la imprimimos por pantalla
# y mostramos una gráfica.
x_train = simula_unif(1000, 2, 1)
print("La muestra de entrenamiento: ")
print(x_train)
plt.scatter(x_train[:,0], x_train[:,1])
plt.title("Ejercicio 2.2.a. Muestra de entrenamiento uniforme")
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2.b ---\n")

# 2.2.b. - Asignar etiquetas dado f(x1, x2) e introducir ruido sobre 10% de las
# mismas.
print("Se muestra gráfica...")

def sign(x):
 	if x >= 0:
         return 1
 	return -1

def f(x1, x2):
 	return sign((x1-0.2)**2 + x2**2 - 0.6)

# Función para introducir ruido aleatoriamente en un vector de etiquetas 'y',
# dada una semilla 'seed'.
def generar_ruido(y, seed):
    rng = np.random.default_rng(seed) # Asignamos a una variable 'rng' un generador de números pseudoaleatorios.
    # Accedemos a posiciones aleatorias y no repetidas (por ello replace=False) durante y.size//10 (10% del tamaño
    # de 'y', truncado) iteraciones y multiplicamos por -1.
    for i in range(0, y.size//10):
        y[rng.choice(y.size, replace=False)] *= -1

# Aquí obtenemos el vector de etiquetas 'y_train' para los datos de entrenamiento 'x_train' y 
# mostramos por pantalla una gráfica con los puntos coloreados según su etiqueta.
y_train = np.array([f(x1, x2) for x1, x2 in x_train], dtype=np.float64)
plt.scatter(x_train[np.where(y_train == 1), 0], x_train[np.where(y_train == 1), 1], c="yellow")
plt.scatter(x_train[np.where(y_train == -1), 0], x_train[np.where(y_train == -1), 1], c="purple")
plt.legend(("+1","-1"), loc="upper right")
plt.title("Ejercicio 2.2.b - Muestra etiquetada sin ruido")
plt.show()

input("\n--- Pulsar tecla para continuar con el ejercicio 2.2.b ---\n")
print("Se muestra gráfica...")

# Introducimos ruido y hacemos otra gráfica con los mismos puntos.
generar_ruido(y_train, seed)
plt.scatter(x_train[np.where(y_train == 1), 0], x_train[np.where(y_train == 1), 1], c="yellow")
plt.scatter(x_train[np.where(y_train == -1), 0], x_train[np.where(y_train == -1), 1], c="purple")
plt.legend(("+1","-1"), loc="upper right")
plt.title("Ejercicio 2.2.b - Muestra etiquetada con ruido")
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2.c ---\n")

# 2.2.c - Ajustar un modelo de regresión lineal al conjunto de datos generado y estimar w con SGD.

# Insertamos una columna de unos a la muestra de entrenamiento 'x_train' y estimamos w con SGD. Imprimimos
# por pantalla el error obtenido.
X_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
w_sgd = sgd(np.array([0.0] * X_train.shape[1], dtype=np.float64), X_train, y_train, eta, error2get, maxIter, 24)
print("Ein: ", Err(X_train, y_train, w_sgd))

# Calculamos los puntos necesarios para obtener la recta de w.
sgd_x = np.linspace(-1, 1, 2)
sgd_y = (-w_sgd[0]-w_sgd[1]*sgd_x)/w_sgd[2]

# Generamos una gráfica con la recta dada por w y los puntos coloreados según su etiqueta.
plt.plot(sgd_x, sgd_y, c="red")
plt.ylim(-1.,1.)
plt.scatter(X_train[np.where(y_train == 1), 1], X_train[np.where(y_train == 1), 2], c="yellow")
plt.scatter(X_train[np.where(y_train == -1), 1], X_train[np.where(y_train == -1), 2], c="purple")
plt.legend(("SGD", "+1", "-1"), loc="upper right")
plt.title("Ejercicio 2.2.c Regresión lineal para la muestra generada")
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio 2.2.d ---\n")

# 2.2.d - Ejecutar el experimento definido de a) a c) 1000 veces, calculando el Ein medio
# para las 1000 muestras y el Eout medio para otras 1000 muestras diferentes.

# Función que añade una columna de unos a una muestra 'x'.
def generar_vectorC_lineal(x):
    return np.hstack((np.ones((x.shape[0], 1)), x))

# Para automatizar el hacer 1000 experimentos, he hecho una función que dado un número
# de iteraciones estima w para una muestra de entrenamiento concreta y para una muestra
# de prueba generada cada iteración usando SGD y calcula el Ein y Eout medio.
# Como argumentos de entrada tiene la muestra 'x_train', el vector de etiquetas 'y' sin ruido,
# el número de iteraciones/experimentos, el máximo de iteraciones para el SGD y un generador
# de vector de características para reutilizar el código en el ejercicio posterior no lineal.
def EinEout_medio(x_train, y_train, iteraciones, maxIter_sgd, generar_vectorC):
    Ein = 0. # Inicializamos el Ein medio a 0.0
    Eout = 0. # Inicializamos el Eout medio a 0.0
    x_train = generar_vectorC(x_train) # Generamos el vector de características para 'x_train'
    
    for i in range(0, iteraciones): # Por cada iteración...
        x = simula_unif(1000, 2, 1) # Se genera una muestra de prueba 'x' con 1000 elementos
        y = np.array([f(x1, x2) for x1, x2 in x], dtype=np.float64) # Se genera el vector de etiquetas 'y' dado 'x'
        x = generar_vectorC(x) # Generamos el vector de características para 'x'
        generar_ruido(y, seed+i) # Introducimos ruido en 'y'. En este caso sí que tenemos que cambiar la semilla
                                 # cada iteración porque default_rng da el mismo valor en cada llamada dada una misma
                                 # semilla, a diferencia de las otras funciones de números aleatorios donde la semilla
                                 # afectaba a un nivel de ejecución.
        
        # Estimamos w para la muestra de entrenamiento.
        w = sgd(np.array([0.0] * x_train.shape[1], dtype=np.float64), x_train, y_train, 0.01, 1e-14, maxIter_sgd, 24)
        # Sumamos a Ein el error para la muestra de entrenamiento dado w.
        Ein += Err(x_train, y_train, w)
        # Sumamos a Eout el error para la muestra de prueba dado el w obtenido con la muestra de entrenamiento.
        Eout += Err(x, y, w)
    # Se devuelve Ein y Eout divido por el número de iteraciones para obtener la media.
    return Ein/iteraciones, Eout/iteraciones

# Calculamos el Ein y Eout medio de 1000 iteraciones dada la muestra de entrenamiento generada
# previamente y para un vector de características lineal. Imprimimos por pantalla el resultado.
Ein_medio, Eout_medio = EinEout_medio(x_train, y_train, 1000, 10, generar_vectorC_lineal)
print("Ein medio: ", Ein_medio)
print("Eout medio: ", Eout_medio)

input("\n--- Pulsar tecla para continuar con el experimento con características no lineales ---\n")

# Repetir el mismo experimento anterior pero usando características no lineales.
# Se utiliza el siguiente vector de características: phi²(x) = (1,x1,x2,x1*x1,x1²,x2²).
# Ajustar el nuevo modelo de regresión lineal y calcular w. Calcular los errores promedio Ein y Eout.

# Función que transforma una muestra 'x' (x1, x2) al vector de características (1,x1,x2,x1*x1,x1²,x2²).
def generar_vectorC_noLineal(x):
    x = np.hstack((np.ones((x.shape[0], 1)), x)) # Se añade una columna de unos al principio de 'x'.
    return np.hstack((x, np.array([(x[:,1]*x[:,2]), (x[:,1]**2), (x[:,2]**2)]).T)) # Se añade el resto de columnas.

# Hacemos el mismo experimento anterior pero para el vector de características no lineal e imprimimos
# por pantalla el resultado.
Ein_medio, Eout_medio = EinEout_medio(x_train, y_train, 1000, 25, generar_vectorC_noLineal)
print("Ein medio: ", Ein_medio)
print("Eout medio: ", Eout_medio)

input("\n--- Pulsar tecla para continuar al ejercicio bonus ---\n")
print("-BONUS.a-\n")

# BONUS: Método de Newton
# Implementar el algoritmo de minimización de Newton para el F(x,y) dado en el ejercicio 1.3.
def d2Fx(x, y):
    return -8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) + 2

def d2Fy(x, y):
    return -8*np.pi**2*np.sin(2*np.pi*x)*np.sin(2*np.pi*y) + 4

def d2Fxy(x, y):
    return 8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

d2Fyx = d2Fxy

def H(x, y):
    return np.array([[d2Fx(x,y), d2Fxy(x,y)], [d2Fyx(x,y), d2Fy(x,y)]], dtype=np.float64)

def metodo_de_newton(initial_point, eta, error2get, maxIter):
    iterations = 0
    w = initial_point
    error = F(w[0], w[1])
    descenso = np.array([iterations, error],dtype=np.float64)
    
    while not abs(error) < error2get and iterations < maxIter:
        w = w - eta*np.matmul(H(w[0], w[1])**-1, gradF(w[0], w[1]))
        error = F(w[0], w[1])
        iterations += 1
        descenso = np.row_stack((descenso, np.array([iterations, error])))
    return w, iterations, descenso

# BONUS.a - Minimizar la función para (x0 = -1, y0 = 1), eta = 0.01 y 50 iteraciones
# como máximo. Repetir para eta = 0.1 .

descenso1_grad = descenso1
descenso2_grad = descenso2

maxIter = 50
initial_point = np.array([-1.0, 1.0], dtype=np.float64)
w1, it1, descenso1 = metodo_de_newton(initial_point, 0.01, error2get, maxIter)
w2, it2, descenso2 = metodo_de_newton(initial_point, 0.1, error2get, maxIter)

print ("Para eta = 0.01\nNúmero de iteraciones: ", it1)
print ("Coordenadas obtenidas: (", w1[0], ", ", w1[1], ")")
print ("Ein: ", F(w1[0], w1[1]))

print ("\nPara eta = 0.1\nNúmero de iteraciones: ", it2)
print ("Coordenadas obtenidas: (", w2[0], ", ", w2[1], ")")
print ("Ein: ", F(w2[0], w2[1]))

plt.plot(descenso1[:,0], descenso1[:,1])
plt.plot(descenso2[:,0], descenso2[:,1], "r")
plt.title("BONUS.a. Método de netwon para F(x,y)")
plt.xlabel("Iteraciones")
plt.ylabel("F(x,y)")
plt.legend(("eta = 0.01", "eta = 0.1"))
plt.show()

input("\n--- Pulsar tecla para continuar al ejercicio BONUS.b ---\n")
print("Se muestra tabla...")

eta = 0.01
a, it, des = metodo_de_newton(np.array([-0.5, -0.5],dtype=np.float64), eta, error2get, maxIter)
b, it, des = metodo_de_newton(np.array([1.0, 1.0],dtype=np.float64), eta, error2get, maxIter)
c, it, des = metodo_de_newton(np.array([2.1, -2.1],dtype=np.float64), eta, error2get, maxIter)
d, it, des = metodo_de_newton(np.array([-3.0, 3.0],dtype=np.float64), eta, error2get, maxIter)
e, it, des = metodo_de_newton(np.array([-2.0, 2.0],dtype=np.float64), eta, error2get, maxIter)
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

plt.title("Ejercicio BONUS.b. Para eta = " +str(eta)+ " y un máximo de " +str(maxIter)+ " iteraciones")
table.scale(2.5,2.5)
plt.show()

input("\n--- Pulsar tecla para continuar a la comparación Grad. descendente / Newton ---\n")
print("Se muestra gráfica...")

plt.plot(descenso1_grad[:,0], descenso1_grad[:,1])
plt.plot(descenso1[:,0], descenso1[:,1], "r")
plt.title("Grad. descendente vs. Newton para eta = 0.01")
plt.xlabel("Iteraciones")
plt.ylabel("F(x,y)")
plt.legend(("Grad. descendente", "Newton"))
plt.show()

input("\n--- Pulsar tecla para continuar a la comparación para eta = 0.1 ---\n")
print("Se muestra gráfica...")

plt.plot(descenso2_grad[:,0], descenso2_grad[:,1])
plt.plot(descenso2[:,0], descenso2[:,1], "r")
plt.title("Grad. descendente vs. Newton para eta = 0.1")
plt.xlabel("Iteraciones")
plt.ylabel("F(x,y)")
plt.legend(("Grad. descendente", "Newton"))
plt.show()