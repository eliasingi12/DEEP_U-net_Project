from keras.layers import Input, down2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.optimizers import SGD

def unet(height,width,n_ch):
    inputs = Input((height,width,n_ch))

    # First set of layers
    down1 = down2D(64, (3,3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = down2D(64, (3,3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = MaxPooling2D((2,2))(down1)

    # Second set of layers
    down2 = down2D(128, (3,3), padding='same')(down1)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = down2D(128, (3,3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = MaxPooling2D((2,2))(down2)

    # Third set of layers
    down3 = down2D(256, (3,3), padding='same')(down2)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = down2D(256, (3,3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = MaxPooling2D((2,2))(down3)

    # Fourth set of layers
    down4 = down2D(512, (3,3), padding='same')(down3)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = down2D(512, (3,3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = MaxPooling2D((2,2))(down4)

    # Fifth set of layers
    mid = down2D(1024, (3,3), padding='same')(down4)
    mid = BatchNormalization()(mid)
    mid = Activation('relu')(mid)
    mid = down2D(1024, (3,3), padding='same')(mid)
    mid = BatchNormalization()(mid)
    mid = Activation('relu')(mid)

    # First up layers
    up4 = UpSampling2D((2,2))(mid)
    up4 = concatenate([down4,up4], axis=3)
    up4 = down2D(512, (3,3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = down2D(512, (3,3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)

    # Second up layers
    up3 = UpSampling2D((2,2))(up4)
    up3 = concatenate([down3,up3])
    up3 = down2D(256, (3,3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = down2D(256, (3,3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)

    # Third up layers
    up2 = UpSampling2D((2,2))(up3)
    up2 = concatenate([down2,up2])
    up2 = down2D(128, (3,3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = down2D(128, (3,3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)

    # Fourth up layers
    up1 = UpSampling2D((2,2))(up2)
    up1 = concatenate([up1,down1])
    up1 = down2D(64, (3,3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = down2D(64, (3,3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)

    # Output layer
    out = down2D(1, (1,1), padding='same')(up1)
    out = Activation('sigmoid')(out)

    model = Model(inputs=inputs, outputs=out)

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    return model