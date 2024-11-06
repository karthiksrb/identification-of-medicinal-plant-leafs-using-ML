from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

model = load_model('keras_model.h5')

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open('leaves/Basella Alba (Basale)/BA-S-001.jpg')
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

prediction = model.predict(data)
y_classes = prediction.argmax(axis=-1)
accu=prediction[0][y_classes]*100
if y_classes==0:
    print("Alpinia with accuracy: ",accu)
elif y_classes==1:
    print("Amaranthus viridis with accuracy: ",accu)
elif y_classes==2:
    print("Artocarpus with accuracy: ",accu)
elif y_classes==3:
    print("Azadirachta viridis with accuracy: ",accu)
elif y_classes==4:
    print("Basella Alba viridis with accuracy: ",accu)
elif y_classes==5:
    print("Brassica Juncea viridis with accuracy: ",accu)