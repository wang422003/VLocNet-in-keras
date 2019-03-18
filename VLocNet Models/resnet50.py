"""ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import print_function

import numpy as np
import warnings
import math

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import concatenate
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.regularizers import l2
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape   #lol _____
from keras.engine.topology import get_source_inputs
from keras import initializers


WEIGHTS_PATH = \
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_' \
    'tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = \
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_' \
    'tf_kernels_notop.h5'
s_x = -3
s_q = -6.5


def euc_lossx_odo(y_true, y_pred):

    l_t_tp = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
    lx = l_t_tp * math.exp(-1*s_x) + s_x
    return lx


def euc_lossq_odo(y_true, y_pred):

    l_t_tp = K.sqrt(K.sum(K.square(y_true[:, :] - y_pred[:, :]), axis=1, keepdims=True))
    lq = l_t_tp * math.exp(-1*s_q) + s_q
    return lq


def geo_loss_interface(pred_odo):
    """The version of using Interface for GEO loss calculation.
    :param pred_odo_pose:
    :param pred_odo_orien:
    :return:
    """
    def geo_loss_det(y_true, y_pred):
        odo_pose = pred_odo[:, 0:3]
        odo_orien = pred_odo[:, 3:]
        geo_pose = 0
        print('In Construction')
    return geo_loss_det


def geo_loss(y_true, y_pred):
    """Loss for the network out put of [[odo_pose, odo_orien, geo_pose, geo_orien], [odo_pose, odo_orien]]
    :param y_true:
    :param y_pred:
    :return:
    """
    odo_pose_pred = y_pred[:, 0:3]
    odo_orien_pred = y_pred[:, 3:7]
    geo_pose_pred = y_pred[:, 7:10]
    geo_orien_pred = y_pred[:, 10:14]

    odo_pose_true = y_true[:, 0:3]
    odo_orien_true = y_true[:, 3:7]
    geo_pose_true = y_true[:, 7:10]
    geo_orien_true = y_true[:, 10:14]

    l_odo_pose = K.sqrt(K.sum(K.square(odo_pose_true - odo_pose_pred), axis=1, keepdims=True))
    l_geo_pose = K.sqrt(K.sum(K.square(geo_pose_true - geo_pose_pred), axis=1, keepdims=True))
    l_odo_orien = K.sqrt(K.sum(K.square(odo_orien_true - odo_orien_pred), axis=1, keepdims=True))
    l_geo_orien = K.sqrt(K.sum(K.square(geo_orien_true - geo_orien_pred), axis=1, keepdims=True))
    loss = (l_odo_orien + l_geo_orien) * math.exp(-1*s_x) + s_x + (l_odo_pose + l_geo_pose) * math.exp(-1*s_q) + s_q
    return loss


def odo_loss(y_true, y_pred):
    """Odo Loss for the network out put of [[odo_pose, odo_orien, geo_pose, geo_orien], [odo_pose, odo_orien]]
    :param y_true:
    :param y_pred:
    :return:
    """
    odo_pose_pred = y_pred[:, 0:3]
    odo_orien_pred = y_pred[:, 3:7]

    odo_pose_true = y_true[:, 0:3]
    odo_orien_true = y_true[:, 3:7]

    l_odo_pose = K.sqrt(K.sum(K.square(odo_pose_true - odo_pose_pred), axis=1, keepdims=True))
    l_odo_orien = K.sqrt(K.sum(K.square(odo_orien_true - odo_orien_pred), axis=1, keepdims=True))
    loss = l_odo_orien * math.exp(-1*s_x) + s_x + l_odo_pose * math.exp(-1*s_q) + s_q
    return loss


def identity_block(input_tensor, kernel_size, filters, stage, block, activation, branch=None):
    """The identity parts in every ResNet modules.
    :param input_tensor: Input
    :param kernel_size:
    :param filters:
    :param stage:
    :param block:
    :param activation:
    :param branch:
    :return:
    """

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch' + branch
    bn_name_base = 'bn' + str(stage) + block + '_branch' + branch

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation(activation)(x)
    # x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation(activation)(x)
    # x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation(activation)(x)
    # x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, activation, strides=(2, 2), branch=None):

    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch' + branch
    bn_name_base = 'bn' + str(stage) + block + '_branch' + branch

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation(activation)(x)
    # x = Activation('relu')(x) # old reset layer

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation(activation)(x)
    # x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation(activation)(x)
    # x = Activation('relu')(x)
    return x


def ResNet_50_unit_1(input_tensor, bn_axis, activation, strides=(2, 2), branch=None):

    x = ZeroPadding2D((3, 3))(input_tensor)
    x = Conv2D(64, (7, 7), strides=strides, name='conv1' + branch)(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1' + branch)(x)
    x = Activation(activation)(x)
    # x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=strides)(x)
    return x


def ResNet_50_unit_2(input_tensor, activation, strides=(1, 1), branch=None):

    x = conv_block(input_tensor, 3, [64, 64, 256], stage=2, block='a', activation=activation, strides=strides,
                   branch=branch)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', activation=activation, branch=branch)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', activation=activation, branch=branch)
    return x


def ResNet_50_unit_3(input_tensor, activation, branch=None):

    x = conv_block(input_tensor, 3, [128, 128, 512], stage=3, activation=activation, block='a', branch=branch)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', activation=activation, branch=branch)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', activation=activation, branch=branch)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', activation=activation, branch=branch)
    return x


def ResNet_50_unit_4(input_tensor, activation, branch=None):

    x = conv_block(input_tensor, 3, [256, 256, 1024], stage=4, block='a', activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', activation=activation, branch=branch)
    return x


def ResNet_50_unit_5(input_tensor, activation, branch=None):

    x = conv_block(input_tensor, 3, [512, 512, 2048], stage=5, block='a', activation=activation, branch=branch)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', activation=activation, branch=branch)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', activation=activation, branch=branch)
    return x


# The defnitio of VLocNet body

def VLocNet(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):  # pooling=None,

    """
    Based on ResNet-50 with the modifications as followed:
    1. Replace the relu activator with the elu activator.
    2. Set the branches for global pose estimation and the visual odometry estimation.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    # First for the Visual Odometry

    odo_1 = ResNet_50_unit_1(input_tensor=img_input[0], bn_axis=bn_axis, activation='elu', strides=(2, 2))

    odo_2 = ResNet_50_unit_2(input_tensor=odo_1, activation='elu', strides=(1, 1))

    odo_3 = ResNet_50_unit_3(input_tensor=odo_2, activation='elu')

    odo_4 = ResNet_50_unit_4(input_tensor=odo_3, activation='elu')

    odo_5 = ResNet_50_unit_5(input_tensor=odo_4, activation='elu')

    avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    glo_ave = GlobalAveragePooling2D()(avg_pool)

    fc_1 = Dense(1024)(glo_ave)

    fc_2 = Dense(3)(fc_1)
    fc_3 = Dense(4)(fc_1)

    '''
    x = ZeroPadding2D((3, 3))(img_input[0])
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('elu')(x)
    #x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', activation='elu', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', activation='elu')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', activation='elu')

    x = conv_block(x, 3, [128, 128, 512], stage=3, activation='elu', block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', activation='elu')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', activation='elu')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', activation='elu')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', activation='elu')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', activation='elu')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', activation='elu')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', activation='elu')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', activation='elu')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', activation='elu')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', activation='elu')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', activation='elu')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', activation='elu')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024)(x)

    x_xyz = Dense(3)(x)
    x_pqwr = Dense(4)(x)
    '''
    '''
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    '''
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(input=inputs, output=[fc_2, fc_3], name='VLocNet')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def VLocNet_Odometry(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):
    # pooling=None,

    """The first version of VLocNet_Odometry part
    Based on ResNet-50 with the modifications as followed:
    1. Replace the relu activator with the elu activator.
    2. Set the branches for global pose estimation and the visual odometry estimation.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # 1st branch for the t-1 odometry regression
    odo_1_0 = ResNet_50_unit_1(input_tensor=img_input[0], bn_axis=bn_axis, activation='elu', strides=(2, 2))

    odo_2_0 = ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1))

    odo_3_0 = ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu')

    odo_4_0 = ResNet_50_unit_4(input_tensor=odo_3_0, activation='elu')

    # 2nd branch for the t odometry regression
    odo_1_1 = ResNet_50_unit_1(input_tensor=img_input[1], bn_axis=bn_axis, activation='elu', strides=(2, 2))

    odo_2_1 = ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1))

    odo_3_1 = ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu')

    odo_4_1 = ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = ResNet_50_unit_5(input_tensor=conca, activation='elu')

    avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    glo_ave = GlobalAveragePooling2D()(avg_pool)

    fc_1 = Dense(1024)(glo_ave)

    fc_2 = Dense(3)(fc_1)
    fc_3 = Dense(4)(fc_1)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(input=inputs, output=[fc_2, fc_3], name='VLocNet')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def VLocNet_Odometry_new(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000): # pooling=None,

    """The second version of VLocNet_Odometry part
    Based on ResNet-50 with the modifications as followed:
    1. Replace the relu activator with the elu activator.
    2. Set the branches for global pose estimation and the visual odometry estimation.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=197,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # if K.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    # 1st branch for the t-1 odometry regression
    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t_p')

    odo_2_0 = ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    odo_3_0 = ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    odo_4_0 = ResNet_50_unit_4(input_tensor=odo_3_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t')

    odo_2_1 = ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    odo_3_1 = ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    odo_4_1 = ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    glo_ave = GlobalAveragePooling2D()(odo_5)

    fc_1 = Dense(1024, name='fc_1')(glo_ave)

    fc_2 = Dense(3, name='fc_2')(fc_1)
    fc_3 = Dense(4, name='fc_3')(fc_1)

    # Create model.
    model = Model(input=[input_odo_0, input_odo_1], output=[fc_2, fc_3], name='VLocNet')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def VLocNet_full(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000): # pooling=None,

    """The first version of VLocNet_full
    Based on ResNet-50 with the modifications as followed:
    1. Replace the relu activator with the elu activator.
    2. Set the branches for global pose estimation and the visual odometry estimation.
    3. Shared layers between the 2nd branch of odo_net and the global pose estimation network: Resnet module 1-3.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=197,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # if K.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    # 1st branch for the t-1 odometry regression
    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t_p')

    odo_2_0 = ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    odo_3_0 = ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    odo_4_0 = ResNet_50_unit_4(input_tensor=odo_3_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t')

    odo_2_1 = ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    odo_3_1 = ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    odo_4_1 = ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1')(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2')(odo_fc_1)
    odo_fc_3 = Dense(4, name='odo_fc_3')(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    pose_4 = ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_geo')

    pose_5 = ResNet_50_unit_5(input_tensor=pose_4, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1')(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2')(pose_fc_1)
    pose_fc_3 = Dense(4, name='pose_fc_3')(pose_fc_1)

    pose_merge = concatenate([odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3], name='pose_merge')  # Modification

    # Create model.
    # model = Model(input=[input_odo_0, input_odo_1], output=[odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3],
    #               name='VLocNet_full')

    # changed the model from 4 outputs to 2 outputs
    model = Model(input=[input_odo_0, input_odo_1], output=[odo_merge, pose_merge], name='VLocNet_full')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def VLocNet_v2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000): # pooling=None,

    """The first version of VLocNet_full
    Based on ResNet-50 with the modifications as followed:
    1. Replace the relu activator with the elu activator.
    2. Set the branches for global pose estimation and the visual odometry estimation.
    3. Shared layers between the 2nd branch of odo_net and the global pose estimation network: Resnet module 1-3.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=197,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # if K.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    # 1st branch for the t-1 odometry regression
    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t_p')

    odo_2_0 = ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    odo_3_0 = ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    odo_4_0 = ResNet_50_unit_4(input_tensor=odo_3_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t')

    odo_2_1 = ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    odo_3_1 = ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    odo_4_1 = ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                     bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                     bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(odo_fc_1)

    odo_fc_3 = Dense(4, name='odo_fc_3', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                     bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    pose_4 = ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_geo')

    # The Previous Pose back-feeding

    input_previous_pose = Input(shape=(7, ), name='input_previous_pose')

    previous_fc_4 = Dense(200704, name='previous_fc_4')(input_previous_pose)

    res_previous = Reshape((14, 14, 1024), name='res_previous')(previous_fc_4)

    # Concatenation the previous pose back to the residual unit
    con_4 = concatenate([pose_4, res_previous], name='previous_and_geo4_merge')

    pose_5 = ResNet_50_unit_5(input_tensor=con_4, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                      bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                      bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(pose_fc_1)

    pose_fc_3 = Dense(4, name='pose_fc_3', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                      bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(pose_fc_1)

    pose_merge = concatenate([odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3], name='pose_merge')  # Modification

    # Create model.
    # model = Model(input=[input_odo_0, input_odo_1], output=[odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3],
    #               name='VLocNet_full')

    # changed the model from 4 outputs to 2 outputs
    model = Model(input=[input_odo_0, input_odo_1, input_previous_pose], output=[odo_merge, pose_merge],
                  name='VLocNet_full')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def VLocNet_v3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000): # pooling=None,

    """The first version of modified VLocNet_full (removed u3 in net)
    Based on ResNet-50 with the modifications as followed:
    1. Replace the relu activator with the elu activator.
    2. Set the branches for global pose estimation and the visual odometry estimation.
    3. Shared layers between the 2nd branch of odo_net and the global pose estimation network: Resnet module 1-3.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=224,
    #                                   min_size=197,
    #                                   data_format=K.image_data_format(),
    #                                   include_top=include_top)
    #
    # if input_tensor is None:
    #     img_input = Input(shape=input_shape)
    # else:
    #     if not K.is_keras_tensor(input_tensor):
    #         img_input = Input(tensor=input_tensor, shape=input_shape)
    #     else:
    #         img_input = input_tensor
    # if K.image_data_format() == 'channels_last':
    #     bn_axis = 3
    # else:
    #     bn_axis = 1

    # 1st branch for the t-1 odometry regression
    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t_p')

    odo_2_0 = ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    # odo_3_0 = ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    odo_4_0 = ResNet_50_unit_4(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, activation='elu', strides=(2, 2), branch='_odo_t')

    odo_2_1 = ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    # odo_3_1 = ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    odo_4_1 = ResNet_50_unit_4(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                     bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                     bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(odo_fc_1)

    odo_fc_3 = Dense(4, name='odo_fc_3', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                     bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    pose_4 = ResNet_50_unit_4(input_tensor=odo_2_1, activation='elu', branch='_geo')

    # The Previous Pose back-feeding

    input_previous_pose = Input(shape=(7, ), name='input_previous_pose')

    previous_fc_4 = Dense(802816, name='previous_fc_4')(input_previous_pose)

    res_previous = Reshape((28, 28, 1024), name='res_previous')(previous_fc_4)

    # Concatenation the previous pose back to the residual unit
    con_4 = concatenate([pose_4, res_previous], name='previous_and_geo4_merge')

    pose_5 = ResNet_50_unit_5(input_tensor=con_4, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                      bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                      bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(pose_fc_1)

    pose_fc_3 = Dense(4, name='pose_fc_3', kernel_initializer=initializers.RandomNormal(mean=0, stddev=0.05),
                      bias_initializer=initializers.RandomNormal(mean=0, stddev=0.05))(pose_fc_1)

    pose_merge = concatenate([odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3], name='pose_merge')  # Modification

    # Create model.
    # model = Model(input=[input_odo_0, input_odo_1], output=[odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3],
    #               name='VLocNet_full')

    # changed the model from 4 outputs to 2 outputs
    model = Model(input=[input_odo_0, input_odo_1, input_previous_pose], output=[odo_merge, pose_merge],
                  name='VLocNet_full')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models',
                                    md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models',
                                    md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='avg_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1000')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


if __name__ == '__main__':
    model = VLocNet(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))