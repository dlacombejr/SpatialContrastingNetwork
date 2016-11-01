import operator
import numpy as np
import theano.tensor as t
from keras import backend as K
from keras.callbacks import Callback
from keras.engine.topology import Layer
from keras.layers import Convolution2D, BatchNormalization, LeakyReLU


def convolution_bn_lr_block(nb_filter=96, nb_row=3, nb_col=3):

    def function(input_):

        net = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, border_mode='same')(input_)
        net = BatchNormalization(mode=0, axis=1)(net)
        net = LeakyReLU()(net)

        return net

    return function


def split_er(x, n_splits=2):

    size = K.shape(x)[0]
    split_size = size / n_splits
    return tuple(
        [x[split_size * split: split_size * split + split_size].reshape((split_size, -1)) for split in xrange(n_splits)]
    )


def spatial_contrasting_loss_pairwise(_, y_predicted):

    # recover samples from concatenated output and flatten
    sample_1, sample_2 = split_er(y_predicted, n_splits=2)

    # get l2 distance of all pairwise comparisons using matrix formulation
    sample_1_ss = K.sum(K.square(sample_1), axis=1, keepdims=True)
    sample_2_ss = K.sum(K.square(sample_2), axis=1)
    samples_dot = K.dot(sample_1, sample_2.T)
    dist = K.sqrt(K.maximum(sample_1_ss - 2 * samples_dot + sample_2_ss, K.epsilon()))

    # convert to exp of negative distance
    exp_neg_dist = K.exp(-dist)

    # extract the diagonal for numerator
    numerator = t.nlinalg.diag(exp_neg_dist)

    # extract two denominators (compare i to j=1:N and compare j to i=1:N)
    denominator_a = K.sum(exp_neg_dist, axis=0)
    denominator_b = K.sum(exp_neg_dist, axis=1)

    # calculate two versions of batch distance ratios
    a = K.mean(-K.log((numerator / (denominator_a + K.epsilon())) + K.epsilon()))
    b = K.mean(-K.log((numerator / (denominator_b + K.epsilon())) + K.epsilon()))

    return (a + b) / 2.  # return average of the two versions


class PatchSampler(Layer):

    def __init__(self, patch_size, batch_size, n_neurons, pairwise=False, **kwargs):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.n_neurons = n_neurons
        self.pairwise = pairwise
        if self.pairwise:
            self.x_new = K.zeros(shape=(self.batch_size, self.n_neurons, self.patch_size, self.patch_size))
        super(PatchSampler, self).__init__(**kwargs)

    def call(self, x, mask=None):
        size = K.shape(x)[3]
        valid_size = size - self.patch_size
        assert t.gt(valid_size, 0)
        if self.pairwise is False:
            indices = K.cast(K.random_uniform((2,), low=0, high=valid_size), dtype='int16')
            x_new = x[:, :, indices[0]:indices[0] + self.patch_size, indices[1]:indices[1] + self.patch_size]
        else:
            indices = (
                t.arange(start=0, stop=self.batch_size, step=1),
                K.cast(K.random_uniform((self.batch_size,), low=0, high=valid_size), dtype='int16'),
                K.cast(K.random_uniform((self.batch_size,), low=0, high=valid_size), dtype='int16')
            )
            x_new = self.x_new
            for i in xrange(self.patch_size):
                for j in xrange(self.patch_size):
                    x_new = t.set_subtensor(
                        x_new[indices[0], :, i, j],
                        x[indices[0], :, indices[1] + i, indices[1] + j]
                    )
        return x_new

    def get_config(self):
        config = {
            'patch_size': self.patch_size,
            'batch_size': self.batch_size,
            'n_neurons': self.n_neurons,
            'pairwise': self.pairwise
        }
        base_config = super(PatchSampler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], input_shape[1], self.patch_size, self.patch_size])


class LrReducer(Callback):

    def __init__(self, patience=0, reduce_rate=0.5, reduce_nb=10, verbose=1, type='acc', mode='max'):
        super(Callback, self).__init__()
        self.patience = patience
        self.wait = 0
        self.reduce_rate = reduce_rate
        self.current_reduce_nb = 0
        self.reduce_nb = reduce_nb
        self.verbose = verbose
        self.type = type
        self.comparison = None
        if mode == 'max':
            self.comparison = operator.gt
            self.best_score = -np.inf
        elif mode == 'min':
            self.comparison = operator.lt
            self.best_score = np.inf
        assert self.comparison

    def on_epoch_end(self, epoch, logs={}):
        current_score = logs.get('val_' + self.type)
        if self.comparison(current_score, self.best_score):
            self.best_score = current_score
            self.wait = 0
            if self.verbose > 0:
                print 'Current best val score: {0}'.format(current_score)
        else:
            if self.wait >= self.patience:
                self.current_reduce_nb += 1
                if self.current_reduce_nb <= 10:
                    lr = self.model.optimizer.lr.get_value()
                    reduced_lr = lr * self.reduce_rate
                    self.model.optimizer.lr.set_value(reduced_lr)
                    print 'Reducing Learning Rate from {0} to {1}'.format(lr, reduced_lr)

                else:
                    if self.verbose > 0:
                        print 'Epoch {0}: early stopping'.format(epoch)
                    self.model.stop_training = True
            self.wait += 1
