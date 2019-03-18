import numpy as np
from keras.optimizers import Adam, sgd, nadam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import resnet50
import SE_resnet50
import odo_data_preparation
import odometry_data
from sklearn.model_selection import train_test_split
import h5py
from adabound import AdaBound
import datetime
import weight_changing
'''
For the training of the VLocNet a.k.a University of Freiburg
on 14:00 23.01.2019 created
'''
graph_path_odo = './Model_Graph/ResNet_odo.png'
graph_path_geo = './Model_Graph/ResNet_geo.png'
graph_path_geo_se = './Model_Graph/ResNet_geo_SE.png'
graph_path_geo_modi = './Model_Graph/ResNet_geo_modi'
graph_path_geo_se_modi = './Model_Graph/ResNet_geo_SE'


def create_model_odo():
    """
    func for the odo model creation
    :return:
    """
    # Load the model and set the optimizer
    # VLocNet odometry model
    print('Loading VLocNet Odo model.........')

    model = resnet50.VLocNet_Odometry_new(weights=None)

    adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)  # clipvalue=1.5
    # 1. SGD SGD = sgd(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True )
    # 2. SGD:
    # SGD = sgd(lr=0.00001, decay=0.001316, momentum=0.9, nesterov=True)
    # Nadam = nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.summary()

    # plot graph
    plot_model(model, to_file=graph_path_odo)
    print('The graph is printed and saved at: ' + graph_path_odo)

    model.compile(optimizer=adam, loss={'fc_2': resnet50.euc_lossx_odo,
                                        'fc_3': resnet50.euc_lossq_odo})

    return model


def create_model_geo():
    """
    func for the odo+geo model creation but without the feedback of previous position into the net
    :return:
    """
    # Load the model and set the optimizer
    # VLocNet Geometry model
    print('Loading VLocNet Geo model.........')

    model = resnet50.VLocNet_full(weights=None)

    adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)  # clipvalue=1.5
    # 1. SGD SGD = sgd(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True )
    # 2. SGD:
    SGD = sgd(lr=0.00001, decay=0.001316, momentum=0.9, nesterov=True)
    # Nadam = nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.summary()

    # plot graph
    plot_model(model, to_file=graph_path_geo)
    print('The graph is printed and saved at: ' + graph_path_geo)

    model.compile(optimizer=adam, loss={'pose_merge': resnet50.geo_loss,
                                        'odo_merge': resnet50.odo_loss})

    return model


def create_model_geo_new():
    """
    func for the odo+geo model creation and includes the feedback of previous position into the net
    :return:
    """
    # Load the model and set the optimizer
    # VLocNet Geometry model
    print('Loading VLocNet Geo model.........')

    model = resnet50.VLocNet_v2(weights=None)

    adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)  # clipvalue=1.5
    # 1. SGD SGD = sgd(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True )
    # 2. SGD:
    # SGD = sgd(lr=0.00001, decay=0.001316, momentum=0.9, nesterov=True)
    # Nadam = nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.summary()

    # plot graph
    plot_model(model, to_file=graph_path_geo)
    print('The graph is printed and saved at: ' + graph_path_geo)

    model.compile(optimizer=adam, loss={'pose_merge': resnet50.geo_loss,
                                        'odo_merge': resnet50.odo_loss})

    return model


def create_model_geo_modi(mode=1):

    print('Loading VLocNet modified Geo model.........')
    if mode == 1:
        st = 'u3'
        model = resnet50.VLocNet_v3(weights=None)
        model.summary()
    else:
        st = ''
        print('Please select the proper model!!')
        model = None
    adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)  # clipvalue=1.5
    # 1. SGD SGD = sgd(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True )
    # 2. SGD:
    SGD = sgd(lr=0.00001, decay=0.001316, momentum=0.9, nesterov=True)
    # 3. adabound
    ada = AdaBound(lr=0.0001,beta_1=0.9, beta_2=0.999,epsilon=0.0000000001 )
    # Nadam = nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    # plot graph
    plot_model(model, to_file=graph_path_geo_modi + st + '.png')
    print('The graph is printed and saved at: ' + graph_path_geo_modi + st + '.png')

    model.compile(optimizer=adam, loss={'pose_merge': resnet50.geo_loss,
                                        'odo_merge': resnet50.odo_loss})

    return model


def create_model_se_geo(mode = 1):
    """
    func for the odo+geo SE model creation and includes the feedback of previous position into the net
    :return:
    """
    # Load the model and set the optimizer
    # SE VLocNet Geometry model
    print('Loading SE-VLocNet Geo model.........')
    if mode == 1:
        model = SE_resnet50.SE_VLocNet_v2(weights=None)
        st = ''
    elif mode == 2:
        # modified SE-VLocNet (no u3 )
        model = SE_resnet50.SE_VLocNet_v3(weights=None)
        st = 'u3'
    else:
        print('Please use the proper SE-VLocNet model!!!!')
        model = None
        st = ''

    adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)  # clipvalue=1.5
    # 1. SGD SGD = sgd(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True )
    # 2. SGD:
    SGD = sgd(lr=0.00001, decay=0.001316, momentum=0.9, nesterov=True)
    # Nadam = nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.summary()

    # plot graph
    plot_model(model, to_file=graph_path_geo_se)
    print('The graph is printed and saved at: ' + graph_path_geo_se_modi + st + '.png')

    model.compile(optimizer=adam, loss={'pose_merge': SE_resnet50.geo_loss,
                                        'odo_merge': SE_resnet50.odo_loss})
    return model


def train_odo_():
    """
    the func for the training with the odo net
    :return:
    """
    # Set the gridsearch for Beta
    # ran = np.linspace(50,2000,((2000-50)//50+1))
    # ran1 = np.linspace(150, 200, ((200-150)//10+1))
    # ran = [50, 100, 150, 200] # 2nd try
    # param_grid = {'Beta': ran}

    # Beta = ran[i]
    checkpointname = 'checkpoint_weights_VLoc_odo' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    save_weight_name = 'custom_trained_weights_VLoc_odo' + str(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    weight_path = './weights/KingsCollege/VLoc/Odo/'
    checkpointpath = weight_path + checkpointname
    save_weight_path = weight_path + save_weight_name

    model = create_model_odo()

    tbCallBack_odo = TensorBoard(log_dir='./Graphs/VLoc/Odo/', histogram_freq=0, write_graph=True, write_images=True)

    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs= -1, return_train_score=False)
    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath=checkpointpath, verbose=1, save_best_only=True, save_weights_only=True)

    # Training proess
    model.fit([X_train_p, X_train], [y_train_x, y_train_q],
              nb_epoch=15000,
              validation_data=([X_valid_p, X_valid], [y_valid_x, y_valid_q]),
              callbacks=[checkpointer, tbCallBack_odo, early_stoping_monitor])
    model.save_weights(save_weight_path)

    print('Weights are saved at: {}'.format(save_weight_path))


def train_geo_():
    """
    the func for the training with the geo net without the feeding of previous position
    :return:
    """
    checkpointname = 'checkpoint_weights_VLoc_geo' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    save_weight_name = 'custom_trained_weights_VLoc_geo' + str(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    weight_path = './weights/KingsCollege/VLoc/Geo/'
    checkpointpath = weight_path + checkpointname
    save_weight_path = weight_path + save_weight_name

    model = create_model_geo()

    tbCallBack_geo = TensorBoard(log_dir='./Graphs/VLoc/Geo/', histogram_freq=0, write_graph=True, write_images=True)

    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath=checkpointpath, verbose=1, save_best_only=True, save_weights_only=True)

    # Training proess
    model.fit([X_train_p, X_train], [y_train_odo, y_train_merge],
              nb_epoch=15000,
              validation_data=([X_valid_p, X_valid], [y_valid_odo, y_valid_merge]),
              callbacks=[checkpointer, tbCallBack_geo, early_stoping_monitor])
    model.save_weights(save_weight_path)

    print('Weights are saved at: {}'.format(save_weight_path))


def train_geo_modi():
    """
    the func for the training with the geo net without the feeding of previous position
    :return:
    """
    mode = 1
    if mode ==1:
        st = 'u3'
    else:
        st =''

    checkpointname = 'checkpoint_weights_VLoc_geo' + st + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) \
                     + '.h5'
    save_weight_name = 'custom_trained_weights_VLoc_geo' + st + str(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    weight_path = './weights/KingsCollege/VLoc/Geo/with_previous/modi/'
    checkpointpath = weight_path + checkpointname
    save_weight_path = weight_path + save_weight_name

    model = create_model_geo_modi(mode= mode)

    tbCallBack_geo = TensorBoard(log_dir='./Graphs/VLoc/Geo/', histogram_freq=0, write_graph=True, write_images=True)

    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath=checkpointpath, verbose=1, save_best_only=True, save_weights_only=True)

    # Training proess
    model.fit([X_train_p, X_train, X_train_previous], [y_train_odo, y_train_merge],
              nb_epoch=15000,
              validation_data=([X_valid_p, X_valid,  X_valid_previous], [y_valid_odo, y_valid_merge]),
              callbacks=[checkpointer, tbCallBack_geo, early_stoping_monitor])
    model.save_weights(save_weight_path)

    print('Weights are saved at: {}'.format(save_weight_path))


def train_geo_new():
    """
    the func for the training with the geo net with the feeding of previous position
    :return:
    """
    checkpointname = 'checkpoint_weights_VLoc_geo_new' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + \
                     '.h5'
    save_weight_name = 'custom_trained_weights_VLoc_geo' + str(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    weight_path = './weights/KingsCollege/VLoc/Geo/with_previous/'
    checkpointpath = weight_path + checkpointname
    save_weight_path = weight_path + save_weight_name

    model = create_model_geo_new()

    tbCallBack_geo = TensorBoard(log_dir='./Graphs/VLoc/Geo/Graph', histogram_freq=0, write_graph=True,
                                 write_images=True)

    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath=checkpointpath, verbose=1, save_best_only=True, save_weights_only=True)

    # Training proess
    model.fit([X_train_p, X_train, X_train_previous], [y_train_odo, y_train_merge],
              nb_epoch=15000,
              validation_data=([X_valid_p, X_valid, X_valid_previous], [y_valid_odo, y_valid_merge]),
              callbacks=[checkpointer, tbCallBack_geo, early_stoping_monitor])
    model.save_weights(save_weight_path)

    print('Weights are saved at: {}'.format(save_weight_path))


def train_geo_se():
    """
    the func for the training with the geo se net with the feeding of previous position
    :return: None
    """
    mode = 2
    if mode == 2:
        st = 'u3'
    else:
        st = ''

    checkpointname = 'checkpoint_weights_VLoc_geo_se' + st+ str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + \
                     '.h5'
    save_weight_name = 'custom_trained_weights_VLoc_geo_se' + st + str(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    weight_path = './weights/KingsCollege/VLoc/se/'
    checkpointpath = weight_path + checkpointname
    save_weight_path = weight_path + save_weight_name

    model = create_model_se_geo(mode=mode)  # 1 for complete SE-VlocNet, 2 for modified SE_VLocNet(no u3)

    tbCallBack_se = TensorBoard(log_dir='./Graphs/VLoc/se/Graph', histogram_freq=0, write_graph=True, write_images=True)

    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath=checkpointpath, verbose=1, save_best_only=True, save_weights_only=True)

    # Training proess
    model.fit([X_train_p, X_train, X_train_previous], [y_train_odo, y_train_merge],
              nb_epoch=15000,
              validation_data=([X_valid_p, X_valid, X_valid_previous], [y_valid_odo, y_valid_merge]),
              callbacks=[checkpointer, tbCallBack_se, early_stoping_monitor])
    model.save_weights(save_weight_path)

    print('Weights are saved at: {}'.format(save_weight_path))


def train_geo_imagenet():
    """
    the func for the training with the geo net with the feeding of previous position
    :return: None
    """
    checkpointname = 'checkpoint_weights_VLoc_geo_IN' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + \
                     '.h5'
    save_weight_name = 'custom_trained_weights_VLoc_geo_IN' + str(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '.h5'
    weight_path = './weights/KingsCollege/VLoc/Geo/with_previous/'
    checkpointpath = weight_path + checkpointname
    save_weight_path = weight_path + save_weight_name

    imagenet_weightpath = './weights/_ImageNet_/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    model = create_model_geo_new()

    model = weight_changing.set_layers_weight(model, imagenet_weightpath) # ,exception_layer_name=['res5', 'bn5']

    tbCallBack_geo = TensorBoard(log_dir='./Graphs/VLoc/Geo/transfer_learning/Graph', histogram_freq=0,
                                 write_graph=True, write_images=True)
    # Setup checkpointing
    checkpointer = ModelCheckpoint(filepath=checkpointpath, verbose=1, save_best_only=True, save_weights_only=True)

    # Training proess
    model.fit([X_train_p, X_train, X_train_previous], [y_train_odo, y_train_merge],
              nb_epoch=15000,
              validation_data=([X_valid_p, X_valid, X_valid_previous], [y_valid_odo, y_valid_merge]),
              callbacks=[checkpointer, tbCallBack_geo, early_stoping_monitor])
    model.save_weights(save_weight_path)

    print('Weights are saved at: {}'.format(save_weight_path))


if __name__ == "__main__":
    # Variables
    batch_size = 8  # origin: 32
    mode = 5  # 1 for odometry, 2 for geo training, 3 for odo and geo(no p_t-1, 4 for odo and geo (with p_t-1)
    # 5 for SE-VlocNet 6 for modified VLocNet 7 for transfer learning for ResNet based VLocNet
    # Load the datasets for training and testing
    dataset_odo_train, dataset_odo_test = odometry_data.get_kings_VLoc_odo(odo_data_preparation.odo_save_train_path,
                                                                           odo_data_preparation.odo_save_test_path)

    # Shape them in forms for the next training
    X_odo_train = np.squeeze(np.array(dataset_odo_train.images))
    X_odo_train_p = np.squeeze(np.array(dataset_odo_train.images_p))
    y_odo_train = np.squeeze(np.array(dataset_odo_train.odo_pose))

    # Split the training labels according for position and orientation
    y_odo_train_x = y_odo_train[:, 0:3]
    y_odo_train_q = y_odo_train[:, 3:7]

    # Shape the labels in form for next testing
    X_odo_test = np.squeeze(np.array(dataset_odo_test.images))
    X_odo_test_p = np.squeeze(np.array(dataset_odo_test.images_p))
    y_odo_test = np.squeeze(np.array(dataset_odo_test.odo_pose))

    y_odo_test_x = y_odo_test[:, 0:3]
    y_odo_test_q = y_odo_test[:, 3:7]

    # Set the reps of early stopping
    early_stoping_monitor = EarlyStopping(patience=500)

    if mode == 1:

        X_train_p, X_valid_p, X_train, X_valid, y_train_x, y_valid_x, y_train_q, y_valid_q = \
            train_test_split(X_odo_train_p, X_odo_train, y_odo_train_x, y_odo_train_q, test_size=0.3, random_state=42)
        print('Training odometry model.......')
        train_odo_()
        print('The training is finished!')

    elif mode == 2:
        print('Loading geometry data ........')

        dataset_geo_train, dataset_geo_test = odometry_data.get_kings_VLoc_geo(odo_data_preparation.save_train,
                                                                               odo_data_preparation.save_test)

        y_geo_train = np.squeeze(np.array(dataset_geo_train.geo_pose))
        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))

        merge_train = np.concatenate((y_odo_train, y_geo_train), axis=1)
        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        X_train_p, X_valid_p, X_train, X_valid, y_train_odo, y_valid_odo, y_train_merge, y_valid_merge = \
            train_test_split(X_odo_train_p, X_odo_train, y_odo_train, merge_train, test_size=0.3, random_state=42)
        print('Geometric data loaded.')
        print('------------------------------')
        print('Training geometry model.......')
        train_geo_()
        print('The training is finished!')

    elif mode == 3:

        print('Loading geometry data ........')

        dataset_geo_train, dataset_geo_test = odometry_data.get_kings_VLoc_geo(odo_data_preparation.save_train,
                                                                               odo_data_preparation.save_test)

        y_geo_train = np.squeeze(np.array(dataset_geo_train.geo_pose))
        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))

        merge_train = np.concatenate((y_odo_train, y_geo_train), axis=1)
        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        X_train_p, X_valid_p, X_train, X_valid, y_train_odo, y_valid_odo, y_train_merge, y_valid_merge = \
            train_test_split(X_odo_train_p, X_odo_train, y_odo_train, merge_train, test_size=0.3, random_state=42)
        print('Geometric data loaded.')
        print('------------------------------')
        print('Training geometry model.......')
        train_geo_()
        print('The training is finished!')

    elif mode == 4:
        print('Loading geometry data ........')

        dataset_geo_train, dataset_geo_test = odometry_data.get_kings_VLoc_geo_new(odo_data_preparation.save_train,
                                                                                   odo_data_preparation.save_test)

        y_geo_train = np.squeeze(np.array(dataset_geo_train.geo_pose))
        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))

        x_geo_previous_train = np.squeeze(np.array(dataset_geo_train.geo_pose_previous))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_train = np.concatenate((y_odo_train, y_geo_train), axis=1)
        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        X_train_p, X_valid_p, X_train, X_valid, X_train_previous, X_valid_previous, \
        y_train_odo, y_valid_odo, y_train_merge, y_valid_merge = \
            train_test_split(X_odo_train_p, X_odo_train, x_geo_previous_train, y_odo_train, merge_train,
                             test_size=0.3, random_state=42)
        print('Geometric data loaded.')
        print('------------------------------')
        print('Training geometry model.......')
        train_geo_new()
        print('The training is finished!')

    elif mode == 5:
        print('Loading geometry data ........')

        dataset_geo_train, dataset_geo_test = odometry_data.get_kings_VLoc_geo_new(odo_data_preparation.save_train,
                                                                                   odo_data_preparation.save_test)

        y_geo_train = np.squeeze(np.array(dataset_geo_train.geo_pose))
        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))

        x_geo_previous_train = np.squeeze(np.array(dataset_geo_train.geo_pose_previous))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_train = np.concatenate((y_odo_train, y_geo_train), axis=1)
        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        X_train_p, X_valid_p, X_train, X_valid, X_train_previous, X_valid_previous, \
        y_train_odo, y_valid_odo, y_train_merge, y_valid_merge = \
            train_test_split(X_odo_train_p, X_odo_train, x_geo_previous_train, y_odo_train, merge_train,
                             test_size=0.3, random_state=42)
        print('Geometric data loaded.')
        print('------------------------------')
        print('Training geometry SE model.......')
        train_geo_se()
        print('The training is finished!')

    elif mode == 6:
        print('Loading geometry data ........')

        dataset_geo_train, dataset_geo_test = odometry_data.get_kings_VLoc_geo_new(odo_data_preparation.save_train,
                                                                                   odo_data_preparation.save_test)

        y_geo_train = np.squeeze(np.array(dataset_geo_train.geo_pose))
        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))

        x_geo_previous_train = np.squeeze(np.array(dataset_geo_train.geo_pose_previous))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_train = np.concatenate((y_odo_train, y_geo_train), axis=1)
        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        X_train_p, X_valid_p, X_train, X_valid, X_train_previous, X_valid_previous, \
        y_train_odo, y_valid_odo, y_train_merge, y_valid_merge = \
            train_test_split(X_odo_train_p, X_odo_train, x_geo_previous_train, y_odo_train, merge_train,
                             test_size=0.3, random_state=42)
        print('Geometric data loaded.')
        print('------------------------------')
        print('Training modified geometry model.......')
        train_geo_modi()
        print('The training is finished!')

    elif mode == 7:
        print('The model will be preweighted with the weight from ImageNet.')
        print('Loading geometry data ........')

        dataset_geo_train, dataset_geo_test = odometry_data.get_kings_VLoc_geo_new(odo_data_preparation.save_train,
                                                                                   odo_data_preparation.save_test)

        y_geo_train = np.squeeze(np.array(dataset_geo_train.geo_pose))
        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))

        x_geo_previous_train = np.squeeze(np.array(dataset_geo_train.geo_pose_previous))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_train = np.concatenate((y_odo_train, y_geo_train), axis=1)
        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        X_train_p, X_valid_p, X_train, X_valid, X_train_previous, X_valid_previous, \
        y_train_odo, y_valid_odo, y_train_merge, y_valid_merge = \
            train_test_split(X_odo_train_p, X_odo_train, x_geo_previous_train, y_odo_train, merge_train,
                             test_size=0.3, random_state=42)
        print('Geometric data loaded.')
        print('------------------------------')
        print('Training geometry model based on the pretrained weights from ImageNet.......')
        train_geo_imagenet()
        print('The training is finished!')

    else:
        print('Please check the MODE value!!!!!!')