from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Model
from keras.applications import VGG16
from keras.layers import Dense
from keras.layers import AveragePooling2D
from keras.layers import Input
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils import paths
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

argpass = argparse.ArgumentParser()
argpass.add_argument("-d", "--data", required=True)
argpass.add_argument("-p", "--plt", default="example.png", type=str)
argpass.add_argument("-m", "--model", default="covid.model", type=str)
arguments = vars(argpass.parse_args())

EPOCHS = 25
BS = 10
INIT_LR = 1e-3

print("[UPDATE] loading....")
path = list(paths.list_images(arguments["data"]))
data = []
lis = []
for p in path:
    img = cv2.imread(p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (220, 220))
    data.append(img)
    lis.append(p.split(os.path.sep)[-2])
data = np.array(data)
lis = np.array(lis)

lis = LabelBinarizer().fit_transform(lis)
lis = to_categorical(lis)
(trainX, testX, trainY, testY) = train_test_split(data, lis, test_size=0.2, stratify=lis, random_state=40)

augmentation = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
base = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(220, 220, 3)))

mainM = base.output
mainM = AveragePooling2D(pool_size=(4, 4))(mainM)
mainM = Flatten(name="flatten")(mainM)
mainM = Dense(64, activation="relu")(mainM)
mainM = Dropout(0.5)(mainM)
mainM = Dense(2, activation="softmax")(mainM)

model = Model(inputs=base.input, outputs=mainM)
for layer in base.layers:
    layer.trainable = False

print("[UPDATING] compiling....")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[UPDATE] training headModel....")
h = model.fit_generator(
    augmentation.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY), validation_steps=len(testX) // BS, epochs=EPOCHS)

print("[UPDATE] evaluate network....")
predId = model.predict(testX, batch_size=BS)
predId = np.argmax(predId, axis=1)

print(classification_report(testY.argmax(axis=1), predId,
                            target_names=LabelBinarizer().classes_))

confmtrx = confusion_matrix(testY.argmax(axis=1), predId)
total = sum(sum(confmtrx))
acc = (confmtrx[0, 0] + confmtrx[1, 1]) / total
sensit = confmtrx[0, 0] / (confmtrx[0, 0] + confmtrx[0, 1])
specif = confmtrx[1, 1] / (confmtrx[1, 0] + confmtrx[1, 1])

print(confmtrx)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensit))
print("specificity: {:.4f}".format(specif))

n = EPOCHS
plt.style.use("ggplot")

plt.figure()
plt.plot(np.arange(0, n), h.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), h.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n), h.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, n), h.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy: ")
plt.title("Accuracy and Training Loss: ")
plt.savefig(arguments["plt"])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save("model.h5")
print("Saved model")
