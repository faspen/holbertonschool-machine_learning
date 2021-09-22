#!/usr/bin/env python3
"""Transfer knowledge"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """Pre-process cifar10 data"""
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    i = K.Input(shape=(32, 32, 3))
    hf = (224 // 32)
    wf = (224 // 32)
    densenet_model = K.layers.Lambda(
        lambda x: K.backend.resize_images(x, height_factor=hf,
                                          width_factor=wf,
                                          data_format='channels_last'))(i)

    DN169 = K.applications.DenseNet169(include_top=False,
                                       weights='imagenet',
                                       input_shape=(224, 224, 3))
    pre_freeze = DN169(densenet_model, training=False)
    pre_freeze = K.layers.GlobalAveragePooling2D(pre_freeze)
    pre_freeze = K.layers.Dense(500, activation='relu')(pre_freeze)
    pre_freeze = K.layers.Dropout(0.2)(pre_freeze)
    output = K.layers.Dense(10, activation='softmax')(pre_freeze)

    model = K.Model(inputs=i, outputs=output)
    DN169.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=K.optimizers.Adam(),
                  metrics=['accuracy'])
    hist = model.fit(x=X_train, y=Y_train,
                     validation_data=(X_test, Y_test),
                     batch_size=300,
                     epochs=5, verbose=True)
    model.save('cifar10.h5')
