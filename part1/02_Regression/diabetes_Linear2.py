from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import numpy as np 
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error, r2_score

d = load_diabetes()
print(d.data, d.target) 
print(d.data.shape)
diabetes_X = d.data[:, np.newaxis, 2]
print(diabetes_X)
      
# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = d.target[:-20]
diabetes_y_test = d.target[-20:]

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
reg = regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
print(reg.score(diabetes_X_test, diabetes_y_test))
# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()      