from sklearn.datasets import load_diabetes
from sklearn import linear_model
import m2cgen as m2c

X, y = load_diabetes(return_X_y=True)

estimator = linear_model.LinearRegression()
estimator.fit(X, y)

code = m2c.export_to_c(estimator)
print(code)
# double score(double * input) {
#     return 152.13348416289597 + input[0] * -10.009866299810744 + input[1] * -239.81564367242345 + input[2] * 519.8459200544603 + input[3] * 324.38464550232385 + input[4] * -792.175638552231 + input[5] * 476.7390210052578 + input[6] * 101.04326793803372 + input[7] * 177.0632376713458 + input[8] * 751.2736995571044 + input[9] * 67.6266921837049;
# }
