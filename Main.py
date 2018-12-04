import pickle
from keras.datasets import fashion_mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
#pobieranie
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
images_train = []
for image_train in x_train:
    images_train.append(image_train.flatten())
images_test = []
for image_test in x_test:
    images_test.append(image_test.flatten())
images_train = np.array(images_train)
images_test = np.array(images_test)

#uczenie
classifier = LogisticRegression(verbose=1, max_iter=10, multi_class="multinomial", solver="sag")
classifier.fit(images_train, y_train)
conf_matrix = confusion_matrix(y_test, classifier.predict(images_test))

#pokaz wyniki
print("Confusion_matrix:")
print(conf_matrix)
sns.heatmap(conf_matrix)
print(classifier.score(images_test, y_test))
#zapisz
pickle.dump(classifier, open('fasion_classifier.model', 'wb'))
