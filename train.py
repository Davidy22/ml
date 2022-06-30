from sklearn.datasets import fetch_openml, load_digits
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from joblib import dump
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib

mnist = fetch_openml("mnist_784")
print(2)
# checking the column names and preprocessing target values in standard format
mnist.keys()
mnist.target = mnist.target.astype(np.int8)#Determining independent and dependent variable and finding the shape
x = np.array(mnist.data)
y = np.array(mnist.target)
x.shape, y.shape
#output ((70000, 784), (70000,))# shuffling the values of x and y
si = np.random.permutation(x.shape[0])
x = x[si]
y = y[si]

some_digit = x[12]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
plt.axis("off")
plt.show()

nca = NeighborhoodComponentsAnalysis(random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)
model = Pipeline([('nca', nca), ('knn', knn)])
model.fit(x,y)
dump(model,"mnist.sav")


print(model.predict(some_digit.reshape(1,-1)))

