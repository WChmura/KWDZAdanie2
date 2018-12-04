import pickle

from keras.preprocessing import image


classifier = pickle.load(open('fasion_classifier.model', 'rb'))
image_file = 'spodnie.jpg'
img = image.load_img(image_file, target_size=(28, 28), grayscale=True)
x = image.img_to_array(img)
print(classifier.predict(x.flatten().reshape(1, -1)))


#wyniki
#koszulka -> t-shirt/top
#koszula -> coat
#spodnie -> spodnie
