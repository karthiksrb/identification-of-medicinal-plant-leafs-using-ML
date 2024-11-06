import tkinter.filedialog
import tensorflow as tf
import tensorflow

from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
# import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tkinter import *
from PIL import Image, ImageTk, ImageOps

model = "keras_model.h5"
model = load_model(model)


data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)


def select_image():
    global panelA, panelB, panelC, panelD
    T.delete('1.0', END)
    path = tkinter.filedialog.askopenfilename()

    if len(path) > 0:
        image = cv2.imread(path)
        image = image_resize(image, width=400, height=200)
        im2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im2 = Image.fromarray(im2)
        im2 = ImageTk.PhotoImage(im2)

        image = Image.open(path)

        size = (224, 224)
        ima = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(ima)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        max_val = np.argmax(prediction, axis=1)

        print(prediction)
        print(max_val)
        per = prediction[0][max_val] * 100
        print("predicted: ", prediction)
        acc = " with accuracy" + str(per)
        if max_val == 0:
            val = "Alpinia" + acc
        elif max_val == 1:
            val = "Amaranthus viridis" + acc
        elif max_val == 2:
            val = "Artocarpus" + acc
        elif max_val == 3:
            val = "Azadirachta viridis" + acc
        elif max_val == 4:
            val = "Basella Alba viridis" + acc
        elif max_val == 5:
            val = "Brassica Juncea viridis" + acc
        else:
            val = "None"
        T.insert(END, str(val))

        if panelA is None:

            panelA = Label(image=im2)
            panelA.image = im2
            panelA.grid(row=3, column=0)

        else:
            panelA.configure(image=im2)
            panelA.image = im2


root = Tk()
root.geometry("800x800")

l = Label(root, text="Predicted: ")
l.config(font=("Courier", 20))
l.grid(row=0, column=0)

T = Text(root, height=5, width=52)
T.grid(row=1, column=0)

Label(root, text="medicinal shop ...").place(x=50, y=50)

panelA = None

btn = Button(root, text="Select an image", command=select_image)
btn.grid(row=2, column=0)

root.mainloop()
