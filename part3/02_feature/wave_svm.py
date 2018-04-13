from sklearn.svm import SVR
import matplotlib.pylab as plt 
import mglearn
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

for gamma in [1, 10]:
    svr = SVR(gamma=gamma).fit(X,y)
    plt.plot(line, svr.predict(line), label='SVR gamma={}'.format(gamma))

plt.plot(X[:,0],y,'o',c='k')
plt.ylabel("regression output")
plt.xlabel("feature")   
plt.show() 

   