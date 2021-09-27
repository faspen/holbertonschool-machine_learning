#!/usr/bin/env python3
"""Transfer knowledge"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """Pre-process cifar10 data"""
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    # Ready the data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    i = K.Input(shape=(32, 32, 3))
    hf = (224 // 32)
    wf = (224 // 32)

    # start with lambda layer
    densenet_model = K.layers.Lambda(
        lambda x: K.backend.resize_images(x, height_factor=hf,
                                          width_factor=wf,
                                          data_format='channels_last'))(i)

    # Use DenseNet169 and initialize weights with imagenet
    DN169 = K.applications.DenseNet169(include_top=False,
                                       weights='imagenet',
                                       input_shape=(224, 224, 3))

    # first instance of freezing
    dNet = DN169(densenet_model, training=False)
    # Go through layer of pooling
    dNet = K.layers.GlobalAveragePooling2D(dNet)
    dNet = K.layers.Dense(500, activation='relu')(dNet)
    dNet = K.layers.Dropout(0.2)(dNet)
    # One more dense layer, 10 filters for the labels
    output = K.layers.Dense(10, activation='softmax')(dNet)

    model = K.Model(inputs=i, outputs=output)
    # Freeze
    DN169.trainable = False

    # Finalize
    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    hist = model.fit(x=X_train, y=Y_train,
                     validation_data=(X_test, Y_test),
                     batch_size=300,
                     epochs=5, verbose=True)
    # save it
    model.save('cifar10.h5')
