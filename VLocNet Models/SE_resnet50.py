"""SE-ResNet-50 model for Keras.
Based on https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings
import math
import numpy as np
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Reshape
from keras.layers import Multiply
from keras.layers import concatenate
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras_applications.imagenet_utils import _obtain_input_shape

WEIGHTS_PATH = \
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_' \
    'tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = \
    'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_' \
    'tf_kernels_notop.h5'
s_x = -3
s_q = -6.5


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


def preprocess_input(x):
    # 'RGB'->'BGR'
    x = x[..., ::-1]

    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68

    # Scale
    x *= 0.017
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block, activation='elu', branch=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001

    block_name = str(stage) + "_" + str(block) + '_' + branch
    conv_name_base = "conv" + '_' + block_name + '_' + branch
    act_name_base = activation + '_' + block_name + '_' + branch

    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    x = Activation(activation=activation, name=act_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    x = Activation(activation=activation, name=act_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)

    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name='fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name='fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])

    x = layers.add([x, input_tensor], name='block_' + block_name)
    x = Activation(activation=activation, name=act_name_base)(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), activation='elu', branch=None):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    bn_eps = 0.0001

    block_name = str(stage) + "_" + str(block) + '_' + branch
    conv_name_base = "conv" + '_'+ block_name + '_' + branch
    act_name_base = activation + '_' + block_name + '_' + branch

    x = Conv2D(filters1, (1, 1), use_bias=False, name=conv_name_base + '_x1')(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x1_bn')(x)
    x = Activation(activation=activation, name=act_name_base + '_x1')(x)

    x = Conv2D(filters2, kernel_size, strides=strides, padding='same', use_bias=False, name=conv_name_base + '_x2')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x2_bn')(x)
    x = Activation(activation=activation, name=act_name_base + '_x2')(x)

    x = Conv2D(filters3, (1, 1), use_bias=False, name=conv_name_base + '_x3')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_x3_bn')(x)

    se = GlobalAveragePooling2D(name='pool' + block_name + '_gap')(x)
    se = Dense(filters3 // 16, activation='relu', name='fc' + block_name + '_sqz')(se)
    se = Dense(filters3, activation='sigmoid', name='fc' + block_name + '_exc')(se)
    se = Reshape([1, 1, filters3])(se)
    x = Multiply(name='scale' + block_name)([x, se])

    shortcut = Conv2D(filters3, (1, 1), strides=strides, use_bias=False, name=conv_name_base + '_prj')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name=conv_name_base + '_prj_bn')(shortcut)

    x = layers.add([x, shortcut], name='block_' + block_name)
    x = Activation(activation=activation, name=act_name_base)(x)
    return x


def SE_ResNet_50_unit_1(input_tensor, bn_axis, bn_eps, activation, strides=(2, 2), branch=None):

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1_' + branch)(input_tensor)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name='conv1_bn_' + branch)(x)
    x = Activation(activation=activation, name='elu1'+branch)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1'+branch)(x)
    return x


def SE_ResNet_50_unit_2(input_tensor, activation, strides=(1, 1), branch=None):

    x = conv_block(input_tensor, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1), activation=activation,
                   branch=branch)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2, activation=activation, branch=branch)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3, activation=activation, branch=branch)
    return x


def SE_ResNet_50_unit_3(input_tensor, activation, branch=None):

    x = conv_block(input_tensor, 3, [128, 128, 512], stage=3, block=1, activation=activation, branch=branch)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2, activation=activation, branch=branch)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3, activation=activation, branch=branch)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4, activation=activation, branch=branch)
    return x


def SE_ResNet_50_unit_4(input_tensor, activation, branch=None):

    x = conv_block(input_tensor, 3, [256, 256, 1024], stage=4, block=1, activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2, activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3, activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4, activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5, activation=activation, branch=branch)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6, activation=activation, branch=branch)
    return x


def SE_ResNet_50_unit_5(input_tensor, activation, branch=None):

    x = conv_block(input_tensor, 3, [512, 512, 2048], stage=5, block=1, activation=activation, branch=branch)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2, activation=activation, branch=branch)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3, activation=activation, branch=branch)
    return x


def SEResNet50(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None,
               pooling=None,
               classes=1000):
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=160,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

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
    bn_eps = 0.0001

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', use_bias=False, name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, epsilon=bn_eps, name='conv1_bn')(x)
    x = Activation('relu', name='relu1')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc6')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='se-resnet50')
    return model


def SE_VLocNet_v1(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):
    # pooling=None,

    """The first version of VLocNet_ without backfeeding of previous pose
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

    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = SE_ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2), branch='_odo_t_p')

    odo_2_0 = SE_ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    odo_3_0 = SE_ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    odo_4_0 = SE_ResNet_50_unit_4(input_tensor=odo_3_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = SE_ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2), branch='_odo_t')

    odo_2_1 = SE_ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    odo_3_1 = SE_ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    odo_4_1 = SE_ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = SE_ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1')(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2')(odo_fc_1)
    odo_fc_3 = Dense(4, name='odo_fc_3')(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    pose_4 = SE_ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_geo')

    # # The Previous Pose back-feeding
    #
    # input_previous_pose = Input(shape=(7, ), name='input_previous_pose')
    #
    # previous_fc_4 = Dense(200704, name='previous_fc_4')(input_previous_pose)
    #
    # res_previous = Reshape((14, 14, 1024), name='res_previous')(previous_fc_4)
    #
    # # Concatenation the previous pose back to the residual unit
    # con_4 = concatenate([pose_4, res_previous], name='previous_and_geo4_merge')

    pose_5 = SE_ResNet_50_unit_5(input_tensor=pose_4, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1')(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2')(pose_fc_1)
    pose_fc_3 = Dense(4, name='pose_fc_3')(pose_fc_1)

    pose_merge = concatenate([odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3], name='pose_merge')  # Modification

    # Create model.
    # model = Model(input=[input_odo_0, input_odo_1], output=[odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3],
    #               name='VLocNet_full')

    # changed the model from 4 outputs to 2 outputs
    model = Model(input=[input_odo_0, input_odo_1], output=[odo_merge, pose_merge],
                  name='VLocNet_full')  #, input_previous_pose

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



def SE_VLocNet_v2(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):
    # pooling=None,

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

    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = SE_ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2), branch='_odo_t_p')

    odo_2_0 = SE_ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    odo_3_0 = SE_ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    odo_4_0 = SE_ResNet_50_unit_4(input_tensor=odo_3_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = SE_ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2), branch='_odo_t')

    odo_2_1 = SE_ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    odo_3_1 = SE_ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    odo_4_1 = SE_ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = SE_ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1')(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2')(odo_fc_1)
    odo_fc_3 = Dense(4, name='odo_fc_3')(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    pose_4 = SE_ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_geo')

    # The Previous Pose back-feeding

    input_previous_pose = Input(shape=(7, ), name='input_previous_pose')

    previous_fc_4 = Dense(200704, name='previous_fc_4')(input_previous_pose)

    res_previous = Reshape((14, 14, 1024), name='res_previous')(previous_fc_4)

    # Concatenation the previous pose back to the residual unit
    con_4 = concatenate([pose_4, res_previous], name='previous_and_geo4_merge')

    pose_5 = SE_ResNet_50_unit_5(input_tensor=con_4, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1')(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2')(pose_fc_1)
    pose_fc_3 = Dense(4, name='pose_fc_3')(pose_fc_1)

    pose_merge = concatenate([odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3], name='pose_merge')  # Modification

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


def SE_VLocNet_v3(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):

    """The third version of full se-VLocNet. Try out on different removal of layers u3 are removed in the odometry
    and geo net
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

    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = SE_ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2), branch='_odo_t_p')

    odo_2_0 = SE_ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    # odo_3_0 = SE_ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    odo_4_0 = SE_ResNet_50_unit_4(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = SE_ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2), branch='_odo_t')

    odo_2_1 = SE_ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    # odo_3_1 = SE_ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    odo_4_1 = SE_ResNet_50_unit_4(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_4_0, odo_4_1], name='conca')

    odo_5 = SE_ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1')(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2')(odo_fc_1)
    odo_fc_3 = Dense(4, name='odo_fc_3')(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    pose_4 = SE_ResNet_50_unit_4(input_tensor=odo_2_1, activation='elu', branch='_geo')

    # The Previous Pose back-feeding

    input_previous_pose = Input(shape=(7, ), name='input_previous_pose')

    previous_fc_4 = Dense(802816, name='previous_fc_4')(input_previous_pose)

    res_previous = Reshape((28, 28, 1024), name='res_previous')(previous_fc_4)

    # Concatenation the previous pose back to the residual unit
    con_4 = concatenate([pose_4, res_previous], name='previous_and_geo4_merge')

    pose_5 = SE_ResNet_50_unit_5(input_tensor=con_4, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1')(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2')(pose_fc_1)
    pose_fc_3 = Dense(4, name='pose_fc_3')(pose_fc_1)

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


def SE_VLocNet_v4(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):

    """The third version of none-previous position se-VLocNet. Try out on different removal of layers u3 are removed
    in the odometry and geo net
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

    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = SE_ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2),
                                  branch='_odo_t_p')

    odo_2_0 = SE_ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    odo_3_0 = SE_ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    # odo_4_0 = SE_ResNet_50_unit_4(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = SE_ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2),
                                  branch='_odo_t')

    odo_2_1 = SE_ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    odo_3_1 = SE_ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    # odo_4_1 = SE_ResNet_50_unit_4(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_3_0, odo_3_1], name='conca')

    odo_5 = SE_ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1')(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2')(odo_fc_1)
    odo_fc_3 = Dense(4, name='odo_fc_3')(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    # pose_4 = SE_ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_geo')  ####

    # The Previous Pose back-feeding

    # input_previous_pose = Input(shape=(7, ), name='input_previous_pose')

    # previous_fc_4 = Dense(802816, name='previous_fc_4')(input_previous_pose)

    # res_previous = Reshape((28, 28, 1024), name='res_previous')(previous_fc_4)

    # Concatenation the previous pose back to the residual unit
    # con_4 = concatenate([pose_4, res_previous], name='previous_and_geo4_merge')

    pose_5 = SE_ResNet_50_unit_5(input_tensor=odo_3_1, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1')(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2')(pose_fc_1)
    pose_fc_3 = Dense(4, name='pose_fc_3')(pose_fc_1)

    pose_merge = concatenate([odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3], name='pose_merge')  # Modification

    # Create model.
    # model = Model(input=[input_odo_0, input_odo_1], output=[odo_fc_2, odo_fc_3, pose_fc_2, pose_fc_3],
    #               name='VLocNet_full')

    # changed the model from 4 outputs to 2 outputs
    model = Model(input=[input_odo_0, input_odo_1], output=[odo_merge, pose_merge],
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


def SE_VLocNet_v5(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, classes=1000):

    """The third version of none-previous position se-VLocNet. Try out on different removal of layers u3 are removed
    in the odometry and geo net
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

    input_odo_0 = Input(shape=(224, 224, 3), name='input_odo_0')

    odo_1_0 = SE_ResNet_50_unit_1(input_tensor=input_odo_0, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2),
                                  branch='_odo_t_p')

    odo_2_0 = SE_ResNet_50_unit_2(input_tensor=odo_1_0, activation='elu', strides=(1, 1), branch='_odo_t_p')

    odo_3_0 = SE_ResNet_50_unit_3(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    # odo_4_0 = SE_ResNet_50_unit_4(input_tensor=odo_2_0, activation='elu', branch='_odo_t_p')

    # 2nd branch for the t odometry regression
    input_odo_1 = Input(shape=(224, 224, 3), name='input_odo_1')

    odo_1_1 = SE_ResNet_50_unit_1(input_tensor=input_odo_1, bn_axis=3, bn_eps=0.0001, activation='elu', strides=(2, 2),
                                  branch='_odo_t')

    odo_2_1 = SE_ResNet_50_unit_2(input_tensor=odo_1_1, activation='elu', strides=(1, 1), branch='_odo_t')

    odo_3_1 = SE_ResNet_50_unit_3(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    # odo_4_1 = SE_ResNet_50_unit_4(input_tensor=odo_2_1, activation='elu', branch='_odo_t')

    # Concatenate the features from 1st and 2nd branches
    conca = concatenate([odo_3_0, odo_3_1], name='conca')

    odo_5 = SE_ResNet_50_unit_5(input_tensor=conca, activation='elu', branch='_odo_all')

    # avg_pool = AveragePooling2D((7, 7), name='avg_pool')(odo_5)

    odo_glo_ave = GlobalAveragePooling2D()(odo_5)

    odo_fc_1 = Dense(1024, name='odo_fc_1')(odo_glo_ave)

    odo_fc_2 = Dense(3, name='odo_fc_2')(odo_fc_1)
    odo_fc_3 = Dense(4, name='odo_fc_3')(odo_fc_1)

    odo_merge = concatenate([odo_fc_2, odo_fc_3], name='odo_merge') # Modification

    # The network branch for the Pose part:

    # pose_4 = SE_ResNet_50_unit_4(input_tensor=odo_3_1, activation='elu', branch='_geo')  ####

    # The Previous Pose back-feeding

    input_previous_pose = Input(shape=(7, ), name='input_previous_pose')

    previous_fc_4 = Dense(401408, name='previous_fc_4')(input_previous_pose)

    res_previous = Reshape((28, 28, 512), name='res_previous')(previous_fc_4)

    # Concatenation the previous pose back to the residual unit

    con_4 = concatenate([odo_3_1, res_previous], name='previous_and_geo4_merge')

    pose_5 = SE_ResNet_50_unit_5(input_tensor=con_4, activation='elu', branch='_geo')

    pose_glo_ave = GlobalAveragePooling2D()(pose_5)

    pose_fc_1 = Dense(1024, name='pose_fc_1')(pose_glo_ave)

    pose_fc_2 = Dense(3, name='pose_fc_2')(pose_fc_1)
    pose_fc_3 = Dense(4, name='pose_fc_3')(pose_fc_1)

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
    model = SE_VLocNet_v2(include_top=True, weights='imagenet')

    img_path = 'elephant.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    print('Predicted:', decode_predictions(preds))