# Finger Counter
A convolutional neural network designed to count how many fingers a hand is holding up in an image and which hand is shown. This model achieved close to 100% accuracy during training.

Technologies used: Python 3, Tensorflow, Keras, Pillow.

# Data

I used a dataset offered for free use on [Kaggle](https://kaggle.com) by the user [koryakinp](https://www.kaggle.com/koryakinp) called [Fingers](https://www.kaggle.com/koryakinp/fingers). It was split into a training and test set. In total, 5/6 of the images were for training and 1/6 were for validation.
The sample hand images were split into 12 different classes, one for each amount of fingers on each hand.

```
['0L', '0R', '1L', '1R', '2L', '2R', '3L', '3R', '4L', '4R', '5L', '5R']
```

![2L](/fingers/test/2L/0a87396a-9e99-4794-b8d3-b3b7f64b9600_2L.png)

# Building the Model

I created a Tensorflow sequential model with 16 total layers. The input layer takes in 128x128 images with a single channel (grayscale). Between each 2D convolutional layer, I used MaxPooling2D and Dropout layers to reduce overfitting. Each of the convolutional layers has 64 filters. I use Flatten along with dense layers in order to give an appropriate model output. The model is not amazingly optimized, but it takes many measures to ensure its accuracy and avoid any overfitting.

```python
model = Sequential()
model.add(Conv2D(64, kernel_size=(2,2), activation='elu', padding='same', input_shape=(img_size, img_size, 1)))

model.add(Conv2D(64, kernel_size=(3,3), activation='elu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv2D(64, kernel_size=(3,3), activation='elu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv2D(64, kernel_size=(3,3), activation='elu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv2D(64, kernel_size=(3,3), activation='elu', padding='same'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(num_classes, activation='softmax'))
```

```python
model.summary()
```
```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 128, 128, 64)      320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 128, 128, 64)      36928     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 64, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 64, 64, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 64, 64, 64)        36928     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 32, 32, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 32, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 32, 32, 64)        36928     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 16, 16, 64)        0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 16, 16, 64)        36928     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 8, 8, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               524416    
_________________________________________________________________
dense_1 (Dense)              (None, 12)                1548      
=================================================================
Total params: 673,996
Trainable params: 673,996
Non-trainable params: 0
_________________________________________________________________
```

# Training the Model

After many different tests and experimentation, I ended up training the model over 5 epochs using a batch size of 96. I used Tensorflow's ImageDataGenerator for data augmentation in order to have a larger quantity of data to test on. The model achieved a very high accuracy by the 5th epoch, with very little loss and an apparent 100% accuracy on the validation data.
```python
model.fit(train_generator, epochs=5, validation_data=validation_generator)
```
```
Epoch 1/6
188/188 [==============================] - 53s 282ms/step - loss: 3.1821 - acc: 0.4849 - val_loss: 0.3251 - val_acc: 0.8667
Epoch 2/6
188/188 [==============================] - 49s 262ms/step - loss: 0.1870 - acc: 0.9334 - val_loss: 0.0854 - val_acc: 0.9689
Epoch 3/6
188/188 [==============================] - 50s 266ms/step - loss: 0.0836 - acc: 0.9722 - val_loss: 0.0229 - val_acc: 0.9928
Epoch 4/6
188/188 [==============================] - 51s 269ms/step - loss: 0.0466 - acc: 0.9850 - val_loss: 0.0123 - val_acc: 0.9958
Epoch 5/6
188/188 [==============================] - 53s 284ms/step - loss: 0.0382 - acc: 0.9877 - val_loss: 0.0012 - val_acc: 1.0000
Epoch 6/6
188/188 [==============================] - 51s 273ms/step - loss: 0.0320 - acc: 0.9899 - val_loss: 7.9302e-04 - val_acc: 1.0000
```

# Testing the Model

I created a small dataset of pictures of my own hand I uploaded from my phone. The model did not fare well and unfortunately had a pretty low accuracy. Given how well it did on the validation data, I was surprised by its performance. After doing some research, it seems likely that the way my phone captures pictures translates poorly to 128x128 rescaling for the purposes of the model. It is also very possible that the nature of the training dataset itself makes it hard to translate to actual images of hands (with part of the arm included).

![My 2R](https://i.gyazo.com/ac127e53d0bcd8bc2c473fe738b233b1.png)

# Conclusion

There were 2 goals I was attempting to reach when creating this model: Learn how to effectively use Tensorflow and count fingers on a hand. This experience helped me get really close at achieving both. I learned a lot about different Python packages useful for image processing and machine learning, including tensorflow, and I managed to reach a validation accuracy of 100% during training. The model's results on real world data, however, definitely had a lot of room to improve.
