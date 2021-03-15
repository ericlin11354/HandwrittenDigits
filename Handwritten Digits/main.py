import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#model = tf.keras.models.load_model('digits.model')

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=5)

test_loss, test_acc = model.evaluate(x=x_test, y=y_test)
#print('\nTest accuracy:', test_acc)

#predictions = model.predict(np.array(x_test))
#print(np.argmax(predictions[1000]))

#plt.imshow(x_test[1000], cmap="gray")
#plt.show()

model.save('digits.model')

'''video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Capturing", gray)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord(' '):
        #cv2.imwrite('image.png', frame)
        predictions = model.predict(np.array(frame).reshape((-1, 784)))
        print(predictions[0])

video.release()

'''

for x in range(1, 6):
    img = cv2.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    #prediction = model.predict(np.invert(np.array([img]).reshape((-1,784))))
    prediction = model.predict(img)
    #prediction = model.predict(np.array(x_test))
    print(np.argmax(prediction))
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
