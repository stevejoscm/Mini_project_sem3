import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


image_dir=r"F:\ric_proj\ric"
Onekg=os.listdir(image_dir+ '\\one_kg')
Twokg=os.listdir(image_dir+ '\\two_kg')
Threekg=os.listdir(image_dir+ '\\three_kg')
Fourkg=os.listdir(image_dir+ '\\four_kg')
Fivekg=os.listdir(image_dir+ '\\five_kg')


print("--------------------------------------\n")

print('The length of One kg images is',len(Onekg))
print('The length of Two kg images is',len(Twokg))
print('The length of Three kg images is',len(Threekg))
print('The length of Four kg images is',len(Fourkg))
print('The length of Five kg images is',len(Fivekg))



print("--------------------------------------\n")
dataset=[]
label=[]
img_siz=(128,128)


for i , image_name in tqdm(enumerate(Onekg),desc="ONE_KG"):
    if(image_name.split('.')[1]=='JPG'):
        image=cv2.imread(image_dir+'/one_kg/'+image_name)   
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
        
for i ,image_name in tqdm(enumerate(Twokg),desc="TWO_KG"):
    if(image_name.split('.')[1]=='JPG'):
        image=cv2.imread(image_dir+'/two_kg/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(0)
        
for i , image_name in tqdm(enumerate(Threekg),desc="THREE_KG"):
    if(image_name.split('.')[1]=='JPG'):
        image=cv2.imread(image_dir+'/three_kg/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(1)
        
        
for i ,image_name in tqdm(enumerate(Fourkg),desc="FOUR_KG"):
    if(image_name.split('.')[1]=='JPG'):
        image=cv2.imread(image_dir+'/four_kg/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)        
        
for i ,image_name in tqdm(enumerate(Fivekg),desc="FIVE_KG"):
    if(image_name.split('.')[1]=='JPG'):
        image=cv2.imread(image_dir+'/five_kg/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize(img_siz)
        dataset.append(np.array(image))
        label.append(2)        
              
dataset=np.array(dataset)
label = np.array(label)
print(label)

print("--------------------------------------\n")
print('Dataset Length: ',len(dataset))
print('Label Length: ',len(label))
print("--------------------------------------\n")


print("--------------------------------------\n")
print("Train-Test Split")
x_train,x_test,y_train,y_test=train_test_split(dataset,label,test_size=0.3,random_state=42)
print("--------------------------------------\n")

print("--------------------------------------\n")
print("Normalaising the Dataset. \n")


#Normalizing the Dataset
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0



#Model building
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Change to 3 neurons for 3 classes
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Change loss function
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

#Model Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {round(accuracy * 100, 2)}')

y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
print('Classification Report:\n', classification_report(y_test, y_pred))



#Load and preprocess a single image
def preprocess_single_image(image_path):
    img_size = (128, 128)
    image = cv2.imread(image_path)
    image = Image.fromarray(image, 'RGB')
    image = image.resize(img_size)
    image = np.array(image)
    image = image.astype('float32') / 255.0
    return image


image_path_to_predict = r"F:\ric_proj\ric\three_kg\SUM06953.JPG"
single_image = preprocess_single_image(image_path_to_predict)

#Reshape the image to fit the model's input shape
single_image = np.expand_dims(single_image, axis=0)

# Make predictions using the model
predictions = model.predict(single_image)
predicted_class = np.argmax(predictions)

class_names = ['less than 3','equal to three' ,'greater than 3']
predicted_label = class_names[predicted_class]

print(f"The predicted label for the image is: {predicted_label}")
