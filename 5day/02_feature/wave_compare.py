from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np 
import mglearn
import matplotlib.pylab as plt

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3,3,1000, endpoint=False).reshape(-1,1)
print(line)

tree = DecisionTreeRegressor(min_samples_split=3).fit(X,y)
plt.plot(line, tree.predict(line), label='decision tree')

reg = LinearRegression().fit(X,y)
plt.plot(line, reg.predict(line), label='regression')

plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel('regression output')
plt.xlabel('feature')
plt.legend(loc = 'best')
plt.show()

#############################################
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot",
                impurity=False, filled=True)

from graphviz import Source
Source.from_file("tree.dot").view()
