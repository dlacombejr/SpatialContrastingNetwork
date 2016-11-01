import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.utils.visualize_util import plot
from utilities.data_utilities import load_data
from keras.layers import Input, MaxPooling2D, merge
from keras.preprocessing.image import ImageDataGenerator
from keras_custom import convolution_bn_lr_block, PatchSampler, spatial_contrasting_loss_pairwise, LrReducer


# define global variables
batch_size = 100
n_pre_train = 50000
nb_epochs_pre_train = 50

# load the data (with no validation split)
(X_train, y_train), (_, __), (X_test, y_test) = load_data(n_train=n_pre_train)

# create fake data for pre-training (not used for loss)
y_train_fake = np.empty((y_train.shape[0], 1, 1, 1))
y_test_fake = np.empty((y_test.shape[0], 1, 1, 1))

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

sample_1 = PatchSampler(patch_size=4, batch_size=batch_size, n_neurons=192, pairwise=True)(net)
sample_2 = PatchSampler(patch_size=4, batch_size=batch_size, n_neurons=192, pairwise=True)(net)

output = merge(inputs=[sample_1, sample_2], mode='concat', concat_axis=0)

model = Model(input=input_, output=[output])
model.compile(optimizer=SGD(lr=0.1, momentum=0.9, nesterov=True), loss=spatial_contrasting_loss_pairwise)
model.summary()
plot(model=model, to_file='./saved/models/spatial_contrasting_network.png')

# define checkpoints
model_checkpoint_path = "./saved/models/spatial_contrasting_network_weights_best.hdf5"
checkpoint = ModelCheckpoint(model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
lr_reducer = LrReducer(patience=3, reduce_rate=0.5, reduce_nb=10, type='loss', mode='min')
callbacks_list = [checkpoint, lr_reducer]

# create data generators
train_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
validation_data_generator = ImageDataGenerator(rescale=1. / 255)

# fit the model
model.fit_generator(
    generator=train_data_generator.flow(X_train, y_train_fake, batch_size=batch_size),
    samples_per_epoch=n_pre_train,
    nb_epoch=nb_epochs_pre_train,
    callbacks=callbacks_list,
    validation_data=validation_data_generator.flow(X_test, y_test_fake, batch_size=batch_size, shuffle=False),
    nb_val_samples=X_test.shape[0]
)
