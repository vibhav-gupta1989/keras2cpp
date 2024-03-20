import tensorflow as tf
from keras.datasets import cifar10
from keras import layers, models, optimizers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

train_images, test_images = train_images/255.0, test_images/255.0


train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)

model = models.Sequential([
    layers.Conv2D(4, (3, 3), input_shape=(32, 32, 3), padding='valid'),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(4, (3, 3), padding='same'),
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(4, (3, 3), padding='same'),
    layers.Flatten(),
    #layers.Dense(64),
    layers.Dropout(0.5),
    layers.Dense(10),
    layers.Activation('softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta')

model.summary()
history = model.fit(train_images, train_labels, epochs=1, batch_size=64, validation_data=(test_images, test_labels))
model.save("my_model_cifar10.keras")

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print('Test accuracy:', test_acc)

print(str(model.predict(test_images[:1])))

with open("./sample_cifar10.dat", "w") as fin:
    fin.write("32 32 3\n")
    a = test_images[0]
    for b in a:
        fin.write('[')
        total = 0
        for c in b:
            
            for d in c:
                total = total + 1
                fin.write(str(d))
                if(total < len(b)-1):
                    fin.write(' ')
                
        fin.write(']\n')


