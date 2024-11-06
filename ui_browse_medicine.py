import tkinter.filedialog
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

import cv2
from tkinter import *
from PIL import Image, ImageOps
from PIL import ImageTk

model = "vgg-rps-final.h5"
model = load_model(model)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

description = "None"


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
        dim = (width, int(h * r))

 
    resized = cv2.resize(image, dim, interpolation=inter)

    return resized


def show_des():
    global description
    print(description)
    if description != "None":
        l7.config(text=description)
        l7.pack()
        l7.place(x=50, y=550)

    else:
        l7.config(text="Please select image first")
        l7.pack()
        l7.place(x=50, y=550)


def select_image():
    
    global panelA, val,description
    
    path = tkinter.filedialog.askopenfilename()

    
    if len(path) > 0:
        image = cv2.imread(path)
        image_dis = image_resize(image, width=400, height=200)
        cv2.imwrite("img.jpeg", image_dis)
       

        image = Image.open(path)

        size = (224, 224)
        ima = ImageOps.fit(image, size, Image.ANTIALIAS)

        image_array = np.asarray(ima)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data)
        max_val = np.argmax(prediction, axis=1)

        print(prediction)
        print(max_val)
        per = prediction[0][max_val] * 100
        print("predicted: ", prediction)

        if max_val == 0:
            print("Alpinia with accuracy: ", per)
            name = "Alpinia Galanga (Rasna) "
            description = "Alpinia galanga,a plant in the ginger family,bears a rhizome used largely as an herb in Unani medicine and as a spice in Arab cuisine and Southeast Asian cookery. It is one of four plants known as galangal.  Its common names include greater galangal, lengkuas, and blue ginger. Its Latin name is Alpinia galanga. The plant grows from rhizomes in clumps of stiff stalks up to 2 metres in height with abundant long leaves that bear red fruit.  This plant's rhizome is the galangal used most often in cookery. It is valued for its use in food and traditional medicine.    The rhizome has a pungent smell and strong taste reminiscent of citrus, black pepper and pine needles.   Red and white cultivars are often used differently, with red cultivars being primarily medicinal, and white cultivars primarily as a spice. The red fruit is used in traditional Chinese medicine and has a flavor similar to cardamom."
        elif max_val == 1:
            print("Amaranthus viridis with accuracy: ", per)
            name = "Amaranthus viridis (Arive-Dantu)"
            description = "Amaranthus viridis is a cosmopolitan species in the botanical family Amaranthaceae and is commonly known as slender amaranth or green amaranth. Amaranthus viridis is an annual herb with an upright, light green stem that grows to about 60–80 cm in height. Numerous branches emerge from the base, and the leaves are ovate, 3–6 cm long, 2–4 cm wide, with long petioles of about 5 cm. The plant has terminal panicles with few branches, and small green flowers with 3 stamens. Green Amaranth can contain up to 38% protein by dry weight.The leaves and seeds contain lysine, an essential amino acid."
        elif max_val == 2:
            print("Artocarpus with accuracy: ", per)
            name = "Artocarpus Heterophyllus (Jackfruit)"
            description = "The jackfruit (Artocarpus heterophyllus), also known as jack tree, is a species of tree in the fig, mulberry, and breadfruit family (Moraceae). Its origin is in the region between the Western Ghats of southern India, all of Bangladesh, Sri Lanka and the rainforests of the Philippines, Indonesia, and Malaysia. The jack tree is well-suited to tropical lowlands, and is widely cultivated throughout tropical regions of the world. It bears the largest fruit of all trees, reaching as much as 55 kg (120 pounds) in weight, 90 cm (35 inches) in length, and 50 cm (20 inches) in diameter.[8][12] A mature jack tree produces some 200 fruits per year, with older trees bearing up to 500 fruits in a year.The edible pulp is 74% water, 23% carbohydrates, 2% protein, and 1% fat. The carbohydrate component is primarily sugars, and is a source of dietary fiber. In a 100-gram (3+1⁄2-ounce) portion, raw jackfruit provides 400 kJ (95 kcal), and is a rich source (20% or more of the Daily Value, DV) of vitamin B6 (25% DV). It contains moderate levels (10-19% DV) of vitamin C and potassium, with no significant content of other micronutrients."
        elif max_val == 3:
            print("Azadirachta Indica with accuracy: ", per)
            name = "Azadirachta Indica (Neem)"
            description = "Azadirachta indica, commonly known as neem, nimtree or Indian lilac,[3] is a tree in the mahogany family Meliaceae. It is one of two species in the genus Azadirachta, and is native to the Indian subcontinent and most of the countries in Africa. It is typically grown in tropical and semi-tropical regions. Neem trees also grow on islands in southern Iran Neem is a fast-growing tree that can reach a height of 15–20 metres (49–66 ft), and rarely 35–40 m (115–131 ft). It is deciduous, shedding many of its leaves during the dry winter months. In March 2020, false claims were circulated on social media in various Southeast Asian countries and Africa, supporting the use of neem leaves to treat COVID-19. The Malaysian Ministry of Health summarized myths related to using the leaves to treat COVID-19, and warned of health risks from over-consumption of the leaves."
        elif max_val == 4:
            print("Basella Alba  with accuracy: ", per)
            name = "Basella Alba (Basil)"
            description = "Basella alba is an edible perennial vine in the family Basellaceae. It is found in tropical Asia and Africa where it is widely used as a leaf vegetable. It is native to the Indian subcontinent, Southeast Asia and New Guinea. Basella alba is a fast-growing, soft-stemmed vine, reaching 10 metres (33 ft) in length. The edible leaves are 93% water, 3% carbohydrates, 2% protein, and contain negligible fat. In a 100 gram reference amount, the leaves supply 19 calories of food energy, and are a rich source (20% or more of the Daily Value) of vitamins A and C,folate, and manganese, with moderate levels of B vitamins and several dietary minerals ."
        elif max_val == 5:
            print("Brassica Juncea  with accuracy: ", per)
            name = "Brassica Juncea(Indian Mustard)"
            description = "Brassica juncea, commonly brown mustard, Chinese mustard, Indian mustard, leaf mustard, Oriental mustard and vegetable mustard, is a species of mustard plant. In a 100-gram (3+1⁄2-ounce) reference serving, cooked mustard greens provide 110 kilojoules (26 kilocalories) of food energy and are a rich source (20% or more of the Daily Value) of vitamins A, C, and K—K being especially high as a multiple of its Daily Value. Mustard greens are a moderate source of vitamin E and calcium. Greens are 92% water, 4.5% carbohydrates, 2.6% protein and 0.5% fat "
        else:
              name = "None"

        image = Image.open("img.jpeg")
        # resized_image = image.resize((300, 205), Image.ANTIALIAS)
        im2 = ImageTk.PhotoImage(image)
        l6.config(text=name)
        if panelA is None:

            l6.pack()
            l6.place(x=
                     550, y=440, anchor="center")
            panelA = Label(image=im2)
            panelA.image = im2
            panelA.place(x=420, y=70)

        else:
            panelA.configure(image=im2)
            panelA.image = im2


root = Tk()
root.geometry("1200x950")

root.title("Leaf Disease Detection")

l1 = Label(root, text="MEDICINAL PLANT LEAVES IDENTIFICATION USING MACHINE LEARNING", font=("Courier", 18, "bold"), fg="Red")
l1.place(x=250, y=20)

l2 = Label(root, text=">>> MENU <<<", font=("Courier", 14, "italic", "bold"), fg="Black")
l2.place(x=60, y=100)

btn = Button(root, text="BROWSE IMAGE", command=select_image, bg="#f9a655", font=("Courier", 14, "italic", "bold"),
             height=1, width=15)
btn.place(x=50, y=150)

l = Label(root, text="DETECTED LEAF", font=("Times New Roman", 14, "bold"), fg="Blue")
l.place(x=550, y=400, anchor="center")

l = Label(root, text="---------------", font=("Courier", 14,  "bold"), fg="Teal")
l.place(x=550, y=420, anchor="center")

l6 = Label(root, text="LEAF", font=("Arial Black", 16, "bold"), fg="Red")
l6.place(x=550, y=440, anchor="center")
l6.place_forget()

l = Label(root, text="---------------", font=("Courier", 14, "italic", "bold"), fg="Teal")
l.place(x=550, y=470, anchor="center")

Button(root, text="SHOW DESCRIPTION", command=show_des, bg="#f9a655", font=("Courier", 14, "italic", "bold"), height=1,
       width=20).place(x=750, y=415)

l7 = Label(root, font=("Times New Roman", 10, "bold"), fg="Maroon",wraplength=1200)
l7.place(x=550, y=550)
l7.place_forget()

panelA = None

root.mainloop()
