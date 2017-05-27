from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.preprocessing.image import NumpyArrayIterator, ImageDataGenerator
import numpy as np
x = np.random.random( size=(100, 1, 200, 300)).astype('float32')
y = np.random.random( size=(100, 3)).astype('float32')
model = Sequential([
	Conv2D( 4, (11,11), activation='relu', padding='same', input_shape=x.shape[1:]),
	MaxPooling2D( (2,2)),
	Conv2D( 8, (7,7), activation='relu', padding='same'),
	MaxPooling2D( (2,2)),
	Conv2D( 16, (5,5), activation='relu', padding='same'),
	MaxPooling2D( (2,2)),
	Conv2D( 32, (3,3), activation='relu', padding='same'),
	MaxPooling2D( (2,2)),
	Conv2D( 64, (3,3), activation='relu', padding='same'),
	MaxPooling2D( (2,2)),
	#Conv2D( 128, (3,3), activation='relu', padding='same'),
	#MaxPooling2D( (2,2)),
	#Conv2D( 256, (3,3), activation='relu', padding='same'),
	#MaxPooling2D( (2,2)),
	Flatten(),
	Dropout(0.05),
	Dense(128, activation='relu'),
	Dropout(0.05),
	Dense(y.shape[1], activation='softmax')
])
print model.summary()
model.compile( loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
#model.fit( x, y)
it = NumpyArrayIterator(x, y, ImageDataGenerator(), 10)
def iterator():
    while True:
        a,b = it.next()
        a += 1
        b += 1
        yield a,b
itr = iterator()
model.fit_generator(itr, steps_per_epoch=10, epochs=20)
