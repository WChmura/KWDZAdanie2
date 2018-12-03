import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import seaborn as sns
import pickle

from sklearn.metrics import confusion_matrix
from keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

images_train =  []
for image_train in x_train:
    images_train.append(image_train.flatten())

images_test = []

for image_test in x_test:
    images_test.append(image_test.flatten())

images_train = np.array(images_train)
images_test = np.array(images_test)

cifar10_classifier = OneVsRestClassifier(LogisticRegression(verbose=1, max_iter=10))
cifar10_classifier.fit(images_train, y_train);
conf_matrix = confusion_matrix(y_test, cifar10_classifier.predict(images_test))
print("Confusion_matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix)
cifar10_classifier.score(images_test, y_test)
pickle.dump(cifar10_classifier, open('multi_class_cifar10_classifier.model', 'wb'));
