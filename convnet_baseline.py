from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot
from utilities.data_utilities import load_data
from keras_custom import convolution_bn_lr_block
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, MaxPooling2D, GlobalAveragePooling2D, Activation


# define global variables
batch_size = 100
n_train = 4000
nb_epochs = 100

# load the data
(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data(n_train=n_train)

# build the model
input_ = Input(shape=(3, 32, 32))

net = convolution_bn_lr_block(nb_filter=96, nb_row=3, nb_col=3)(input_)
net = convolution_bn_lr_block(nb_filter=96, nb_row=3, nb_col=3)(net)
net = convolution_bn_lr_block(nb_filter=96, nb_row=3, nb_col=3)(net)

net = MaxPooling2D(pool_size=(2, 2))(net)

net = convolution_bn_lr_block(nb_filter=192, nb_row=3, nb_col=3)(net)
net = convolution_bn_lr_block(nb_filter=192, nb_row=3, nb_col=3)(net)
net = convolution_bn_lr_block(nb_filter=192, nb_row=3, nb_col=3)(net)

net = MaxPooling2D(pool_size=(2, 2))(net)

net = convolution_bn_lr_block(nb_filter=192, nb_row=3, nb_col=3)(net)
net = convolution_bn_lr_block(nb_filter=192, nb_row=1, nb_col=1)(net)
net = convolution_bn_lr_block(nb_filter=10, nb_row=1, nb_col=1)(net)

net = GlobalAveragePooling2D()(net)

net = Activation('softmax')(net)

model = Model(input=input_, output=[net])
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
plot(model=model, to_file='./saved/models/convnet_baseline.png')

# checkpoint
file_path = "./saved/models/convnet_baseline_weights_best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# create data generators
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
valid_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# fit the model
model.fit_generator(
    generator=train_datagen.flow(X_train, y_train, batch_size=batch_size),
    samples_per_epoch=n_train,
    nb_epoch=nb_epochs,
    callbacks=callbacks_list,
    validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=batch_size, shuffle=False),
    nb_val_samples=X_valid.shape[0]
)

# test the model
test_loss, test_acc = model.evaluate_generator(
    generator=test_datagen.flow(X_test, y_test, batch_size=batch_size, shuffle=False),
    val_samples=X_test.shape[0]
)
print 'Test loss: {0} | Test accuracy: {1}'.format(test_loss, test_acc)
