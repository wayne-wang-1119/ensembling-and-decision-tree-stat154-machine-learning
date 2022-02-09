from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

##ensembling method of boosted classification tree
def btc(T, X, y, D):
    n = X.shape[0]
    f = []
    w = [1/n]*n
    e = []
    alpha = []
    

    for t in range(T):  
        dtc = DecisionTreeClassifier(max_depth=D)

        f.append(dtc.fit(X, y, sample_weight=w))
        out =[]
        #for i in range(n):
        #    out.append([1.0 if f[t].predict(X[i,].reshape(1, -1), check_input=True) != y[i] else 0.0][0])
        out = [f[t].predict(X, check_input=True) != y]
        
        e.append(np.dot(np.array(out),np.array(w))/np.sum(w))

        alpha.append((math.log((1-e[t])/e[t]))/2)
        
        for i in range(n):
            w[i] = w[i]*math.exp(alpha[t]*[1.0 if f[t].predict(X[i,].reshape(1, -1), 
                                                                   check_input=True) != y[i] else 0.0][0])      
    def f_ens(x):      
        return np.sign(sum([alpha[t]*f[t].predict(x, check_input=True) for t in range(T)]))

    return f_ens


##testing the tree function defined above
X = pd.read_csv("mushrooms_X.csv").to_numpy()
y = pd.read_csv("mushrooms_Y.csv").to_numpy()
X = X[:,1:]
y = y[:,1]

##sample splitting
test_ix = np.random.choice(X.shape[0], size = int(0.25*X.shape[0]), replace=False)
train_ix = [i for i in range(X.shape[0]) if i not in test_ix]
X_test, y_test = X[test_ix], y[test_ix]
X_train, y_train = X[train_ix], y[train_ix]
T_list = list(range(0,100,5))
T_list[0] = 1
T_list
D = 2
accu = []
for t in T_list:
    accu.append(np.sum([y_test == btc(t, X_train, y_train,D)(X_test)])/X_test.shape[0])
print(accu)
##report accuracy to see how good the tree model is