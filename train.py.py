from keras.layers import Input, Lambda, Dense, Flatten, Dropout
from keras.models import Model
from keras.applications.vgg19 import VGG19
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# In[2]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

# In[3]:


train_path = "dataset/train"
test_path = "dataset/test"
val_path = "dataset/val"

# In[4]:


x_train = []

for folder in os.listdir(train_path):
    sub_path = train_path + "/" + folder
    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_train.append(img_arr)

# In[5]:


x_test = []

for folder in os.listdir(test_path):
    sub_path = test_path + "/" + folder
    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_test.append(img_arr)

# In[6]:


x_val = []

for folder in os.listdir(val_path):
    sub_path = val_path + "/" + folder
    for img in os.listdir(sub_path):
        image_path = sub_path + "/" + img
        img_arr = cv2.imread(image_path)
        img_arr = cv2.resize(img_arr, (224, 224))
        x_val.append(img_arr)

# In[7]:

#convert to matrix
train_x = np.array(x_train)
test_x = np.array(x_test)
val_x = np.array(x_val)

# In[8]:


train_x.shape, test_x.shape, val_x.shape

# In[9]:

train_x = train_x / 255.0
test_x = test_x / 255.0
val_x = val_x / 255.0

# In[10]:


from keras.preprocessing.image import ImageDataGenerator

# In[11]:



train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=(224, 224),
                                                 batch_size=32,
                                                 class_mode='sparse')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(224, 224),
                                            batch_size=32,
                                            class_mode='sparse')

val_set = val_datagen.flow_from_directory(val_path,
                                          target_size=(224, 224),
                                          batch_size=32,
                                          class_mode='sparse')

# In[12]:


print(training_set.class_indices)

# In[13]:


train_y = training_set.classes

# In[14]:


test_y = test_set.classes

# In[15]:


val_y = val_set.classes

# In[16]:


print(train_y.shape, test_y.shape, val_y.shape)

vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(6, activation='softmax')(x)



model = Model(inputs=vgg.input, outputs=prediction)

model.summary()



model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer="adam",
    metrics=['accuracy']
)



from keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

history = model.fit(
    train_x,
    train_y,
    validation_data=(val_x, val_y),
    epochs=2,
    callbacks=[early_stop],
    batch_size=2, shuffle=True)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()

plt.savefig('vgg-loss-rps-1.png')
plt.show()

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()

plt.savefig('vgg-acc-rps-1.png')
plt.show()

# In[26]:


model.evaluate(test_x, test_y, batch_size=32)

# In[27]:


# In[36]:



model.save("vgg-rps-final.h5")

# In[ ]:




