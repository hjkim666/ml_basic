from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)

from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["cancer", "normal"],
                feature_names=cancer.feature_names, impurity=False, filled=True)

from graphviz import Source
Source.from_file("tree.dot").view()

####################################################################
print("feature importance:\n{}".format(tree.feature_importances_))

import matplotlib.pylab as plt
import numpy as np 

def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.figure(figsize=(10,4))
    plt.subplots_adjust(left=0.2)
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("feature importance")
    plt.ylabel("feature")
    plt.show() 
    
plot_feature_importances_cancer(tree)

