
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
import cv2 
import numpy as np
def load_data():
    train_data =  tf.keras.utils.image_dataset_from_directory(
    './dataset/train',  # relative path from your script
    image_size=(48, 48),
    batch_size=32,  #take a batch of size in data contains 32 img
    label_mode="categorical",
    shuffle=True,
    color_mode="grayscale"
    )
    test_data= tf.keras.utils.image_dataset_from_directory(
    './dataset/test',
    image_size=(48,48),
    label_mode="categorical",
    color_mode="grayscale",
    )
    return train_data, test_data

def training():
    train_data =  tf.keras.utils.image_dataset_from_directory(
    './dataset/train',  # relative path from your script
    image_size=(48, 48),
    batch_size=32,  #take a batch of size in data contains 32 img
    label_mode="categorical",
    shuffle=True,
    color_mode="grayscale"
     )
    test_data= tf.keras.utils.image_dataset_from_directory(
    './dataset/test',
    image_size=(48,48),
    label_mode="categorical",
    color_mode="grayscale",   
    )
    model= tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation="relu",input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation="softmax")
     ])
    model.compile(
    optimizer="adam",
    loss= tf.keras.losses.CategoricalCrossentropy(),
    metrics= ["accuracy"]
    )
    early_stop= EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode=min,
    restore_best_weights=True
    )
    checkpoint_callback= keras.callbacks.ModelCheckpoint(
         'best_model.keras',
          monitor= 'val_loss',
          mode='min',
          save_best_only=True
    )
    history= model.fit(
    train_data,
    epochs=1,
    validation_data= test_data,
    callbacks=[checkpoint_callback, early_stop]
    )
   
    return history

def detect_face(img):
     # facecascade= cv2.CascadeClassifier('haarcascadefrontalface_default.xml')
     facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
     gracy_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
     
     face= gracy_img.copy()
     #detectMultiScale :input=> is an img output=> is a rectangle
     face_rectangle= facecascade.detectMultiScale(face, scaleFactor=1.2, minNeighbors=5)
     #nbr of faces found
     print(len(face_rectangle))
     #(x, y) = (50, 100) → top-left corner of the rectangle
     #(x + w, y + h) = (50+80, 100+80) = (130, 180) → bottom-right corner
     #(255, 255, 255) → white color
     #10 → thickness of the rectangle
     model= tf.keras.models.load_model('best_model.keras')
     for (x,y, w, h) in face_rectangle:
          cv2.rectangle(img, (x,y), (x + w, y + h), (0,128,0),10)
          image_extraction= gracy_img[y:y+h, x: x+w] #y to y+h / gracy_img(height, width)
          resize_extracted_face= cv2.resize(image_extraction, (48,48), interpolation= cv2.INTER_LINEAR)
          resheapee= np.reshape(resize_extracted_face,(1,48,48,1)) #only one img
          #32 grayscale images of 48×48 pixels
          print("test")
          predict_val= model.predict(resheapee)
          index_emotion = np.argmax(predict_val)
          class_names= ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
          emotion= class_names[index_emotion] 
          print(emotion)
          img= cv2.putText(img, emotion,(x,h), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0),3,cv2.LINE_AA)
          plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
          plt.show()
     return img

# load_data()
# training()
# src = cv2.imread(r'C:\Users\hp\Downloads/frt.jpg')
# face,predict_val = detect_face(src)
# plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
# plt.axis("off")
# plt.show()


#test a video
from cv2 import VideoCapture, imshow, waitKey, destroyAllWindows
capture= VideoCapture(0)
if not capture.isOpened():
     print("tha camera connection is not established")
     capture.model
while capture.isOpened():
     ret, frame = capture.read()
     if ret:
         imshow('displaying establishing connection',frame)
         _, frame = capture.read()
         detect_face(frame)

     if waitKey(25) ==27:
          break
capture.release()
destroyAllWindows()