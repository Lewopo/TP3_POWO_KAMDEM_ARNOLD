import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0) 
# n_samples = 200
# x = np.linspace(0, 10, n_samples).reshape((n_samples, 1))
# y = x + np.random.randn(n_samples, 1)
# #plt.scatter(x, y) #le résultat: X abscisse et y en ordonnée
# #plt.show()
# # ajout de la colonne de biais a X
# X = np.hstack((x, np.ones(x.shape)))
# print(X.shape)
# # création d'un vecteur parametre theta
# theta = np.random.randn(2, 1)
# print(theta.shape)

def model(X, theta):
    return X.dot(theta)
def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m) * np.sum((model(X, theta) - y)**2)
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)

def gradient_descent(X, y, theta, learning_rate, n_iterations):
# un tableau pour enregistrer l'évolution du Cout du modele
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta) # (formule du gradient descent)
        cost_history[i] = cost_function(X, y, theta) # on enregistre la valeur du Cout au tour i dans cost_history[i]
    return theta, cost_history

# n_iterations = 1000
# learning_rate = 0.01
# theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
# print(theta_final) #parametres du modele une fois que la machine a été entrainée
# # création d'un vecteur prédictions qui contient les prédictions de notre modele final
# predictions = model(X, theta_final)
# # les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
# plt.scatter(x, y)
# plt.plot(x, predictions, c='r')
# plt.show()

# plt.plot(range(n_iterations), cost_history)
# plt.show()

def coef_determination(y, previsions):
    dividante = ((y - previsions)**2).sum()
    diviseur = ((y - y.mean())**2).sum()
    return 1 - (dividante/diviseur)
