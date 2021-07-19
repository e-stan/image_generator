from dataset_helper import *
import matplotlib.pyplot as plt
fn = "../data/abstract_images/0.jpg"
image = readImage("../data/abstract_images/0.jpg")
image = enhanceImage(image,enhancement=8,method="NN")
saveImage(image,"../data/abstract_images/0.jpg".replace(".jpg","_enhanced.jpg"))
plt.figure()
plt.imshow(image)
plt.show()