from sklearn.datasets import load_diabetes
from sklearn import linear_model
import m2cgen as m2c

X, y = load_diabetes(return_X_y=True)

estimator = linear_model.LinearRegression()
estimator.fit(X, y)

code = m2c.export_to_c(estimator)