##A basic model for movie recommendation .
##Here only 6 different movies are taken into account and there are only 5 users .
##The ratings are scaled between 1.0 - 5.0 .
##It is not necessary that a user has rated all the movies . Some ratings may be missing .
##Collaborative filtering approach .
##In the dataset where the user didn't rated a movie a sentinel value has been included explicitly (Here 10.0).
##Here the movies have two features : Romance and action only.
##Here we will predict which movie to watch next out of these 6 movies .
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
alpha = 0.01   ##Learning rate
lamda = 0.01   ##For regularisation
no_of_movies = 6
no_of_users = 5
epochs = 10000  ##No of iterations for gradient Descent to converge at optimal point.

def changerfunc(X,Y,hypothesis,Theta):   ##Extension of GradientDescent function.
    loss = hypothesis
    for i in range(no_of_users):
        for j in range(no_of_movies):
            if(Y[j][i] < 10.0):
                loss[j][i] -= Y[j][i]
            else:
                loss[j][i] = 0.0
    temp1 = alpha*(np.matmul(loss,Theta) + lamda*(X))
    temp2 = alpha*(np.matmul(np.transpose(loss),X) + lamda*(Theta))
    X = X - temp1
    Theta = Theta - temp2
    return X,Theta

def costfunc(X,Y,Theta):
    J = 0.0
    for i in range(no_of_movies):
        for j in range(no_of_users):
            hypothesis = np.matmul(X,np.transpose(Theta))  ##hypothesis should be of same size as of Y which will be...
                                                           ##...achieved by multiplying in this way .
            if Y[i][j]<=5.0:                               ##Removing those entries where there is no rating.
                J += ((1/2)*math.pow((hypothesis[i][j] - Y[i][j]),2))    ##This is loss term or least squared error .
    ##Now we add regularisation term to avoid overfitting .
    ##Firstly adding regularisation term because of X.
    for i in range(no_of_movies):
        for j in range(2):
            J += (lamda/2)*math.pow(X[i][j],2)
    ##Now add regularisation term because of Theta matrix.
    for i in range(no_of_users):
        for j in range(2):
            J += (lamda/2)*math.pow(Theta[i][j],2)
    ##Here one thing should be noticed that we have taken account of both X and Theta matrix in a same cost function.
    ##We can do this.As the loss term in the expression of both X and Theta are same so we keep it once and added...
    ##...regularisation term of both matrices.
    return float(J)

def GradientDesc(X,Theta,Y):
    cost = []
    iters = []
    for i in range(epochs):
        J = costfunc(X,Y,Theta)                                ##Function to compute cost.
        hypothesis = np.matmul(X, np.transpose(Theta))         ##Our hypothesis in linear regression .
        cost.append(J)
        iters.append(i)
        X,Theta = changerfunc(X,Y,hypothesis,Theta)
    plt.plot(iters,cost,'-',color = 'b')
    plt.show()
    return X,Theta

def RandInit():
    X = np.random.uniform(low=0.0, high=2.0, size=(6,2) )
    Theta = np.random.uniform(low = 0 ,high = 5,size = (5,2))
    return X,Theta

def main():
    data = pd.read_csv('ratings.csv',header = None,usecols = range(0,6)).values  ##Importing data
    Y = data[1:]
    Y = np.delete(Y,0,1)   ##Forming the matrix of ratings by users .
    for i in range(no_of_movies):
        for j in range(no_of_users):
            Y[i][j] = float(Y[i][j])
    ##In matrix 10.0 denotes a sentinel value which means that user hasn't rated that movie.
    X,Theta = RandInit()   ##Here X is a matrix which denotes that what fraction of movie is romance and what part of it
                           ##is action. Theta denotes that how much that a user likes romance movie and how much they
                           ##...like an action movie.These lie between 0 - 5 .
    ##Now X and transpose(Theta) should be equal to Y.
    ##Now we will get the X and Theta matrix using supervised learning . We will use linear regression to predict...
    ##...the similar movies and X and Theta.
    X,Theta = GradientDesc(X,Theta,Y)
    print(X)
    print(Theta)

    print("Prediction is:")
    print(np.matmul(X,np.transpose(Theta)))
main()
