import h5py
import numpy as np
from random import randint
from tqdm import tqdm


def isGroup(obj):
    if isinstance(obj, h5py.Group):
        return True
    return False


def isDataset(obj):
    if isinstance(obj, h5py.Dataset):
        return True
    return False


def getDatasetFromGroup(datasets, obj):

    if isGroup(obj):
        for key in obj:
            x = obj[key]
            getDatasetFromGroup(datasets, x)
    else:
        datasets.append(obj)


def getWeightsForLayer(layerName, fileName):

    weights = []
    with h5py.File(fileName, mode='r') as f:
        for key in f:
            if layerName == key: # changed from 'in' to '=='
                datasets = []
                obj = f[key]
                getDatasetFromGroup(datasets, obj)

                for dataset in datasets:
                    w = np.array(dataset)
                    weights.append(w)
    return weights


def set_layers_weight(model, model_path, exception_layer_name= '__'):
    """Set the weight of VLocNet-geo according to the pretrained weight of ResNet50 on ImageNet and freeze the layers in
    training
    default function for weight initialising of VLocNet which based on ResNet50
    :param model: the model of VLocNet-geo with backfeeding of previous position
    :param model_path: config: imagenet weight for resnet50
    :return: the model with set weights
    """
    layers = [layer for layer in model.layers]
    print('Seting weight............')
    for i in tqdm(range(len(layers))):

        layer = layers[i]
        if '_odo_t_p' in layer.name:
            namecache = layer.name.replace('_odo_t_p', '')
            ima_weight = getWeightsForLayer(namecache, model_path)

            layer.set_weights(ima_weight)
            # freeze_layer_trainable(layer, exception_layer_name)
        elif ('_odo_t' in layer.name) and ('_odo_t_p' not in layer.name):
            namecache = layer.name.replace('_odo_t', '')
            ima_weight = getWeightsForLayer(namecache, model_path)

            layer.set_weights(ima_weight)
            # freeze_layer_trainable(layer, exception_layer_name)
        elif '_odo_all' in layer.name:
            namecache = layer.name.replace('_odo_all', '')
            ima_weight = getWeightsForLayer(namecache, model_path)
            if 'res5a' in layer.name:
                if ('_all2b' in layer.name) or ('_all2c' in layer.name):

                    layer.set_weights(ima_weight)

                elif ('_all2a' in layer.name) or ('_all1' in layer.name):
                    ima_weight_new = []
                    for w in ima_weight:
                        if w.ndim == 1:
                            ima_weight_new.append(w)
                        elif w.ndim == 4:
                            ima_weight_new.append(np.concatenate((w, w), axis=2))

                    layer.set_weights(ima_weight_new)

                else:
                    layer.set_weights(ima_weight)
            else:

                layer.set_weights(ima_weight)
            # freeze_layer_trainable(layer, exception_layer_name)

        elif '_geo' in layer.name:
            namecache = layer.name.replace('_geo', '')
            ima_weight = getWeightsForLayer(namecache, model_path)
            if 'res4' in layer.name:

                layer.set_weights(ima_weight)

            elif 'res5a' in layer.name:
                if ('geo2a' in layer.name) or ('_geo1' in layer.name):
                    ima_weight_new = []
                    for w in ima_weight:

                        if w.ndim == 1:
                            ima_weight_new.append(w)

                        elif w.ndim == 4:
                            ima_weight_new.append(np.concatenate((w, w), axis=2))

                    layer.set_weights(ima_weight_new)
                else:

                    layer.set_weights(ima_weight)
            else:

                layer.set_weights(ima_weight)
            # freeze_layer_trainable(layer, exception_layer_name)

    return model


def check_layer_weight(model, model_path):
    """
    Check whether the weight is properly loaded (use samples) DEMO, not suggest to use it.
    :param model:
    :param model_path:
    :return:
    """
    layers = [layer for layer in model.layers]
    for i in tqdm(range(10)):
        lay = layers[randint(0, len(layers) - 1)]
        if '_odo_t_p' in lay.name:
            namecache = lay.name.replace('_odo_t_p', '')
            ima_weight = np.array(getWeightsForLayer(namecache, model_path))

            wei = np.array(lay.get_weights())
            print(lay.name)
            print(np.array_equal(ima_weight, wei))
            if not np.array_equal(ima_weight, wei):
                print('wei:')
                print(wei)
                for ii in range(len(wei)):

                    print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                print('ima_weight')
                print(ima_weight)
                for ii in range(len(ima_weight)):

                    print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))

        elif ('_odo_t' in lay.name) and ('_odo_t_p' not in lay.name):
            namecache = lay.name.replace('_odo_t', '')
            ima_weight = np.array(getWeightsForLayer(namecache, model_path))

            wei = np.array(lay.get_weights())
            print(lay.name)
            print(np.array_equal(ima_weight, wei))
            if not np.array_equal(ima_weight, wei):
                print('wei:')
                print(wei)
                for ii in range(len(wei)):

                    print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                print('ima_weight')
                print(ima_weight)
                for ii in range(len(ima_weight)):

                    print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))

        elif '_odo_all' in lay.name:
            namecache = lay.name.replace('_odo_all', '')
            ima_weight = np.array(getWeightsForLayer(namecache, model_path))
            if 'res5a' in lay.name:
                if ('_all2b' in lay.name) or ('_all2c' in lay.name):
                    wei = np.array(lay.get_weights())
                    print(lay.name)
                    print(np.array_equal(ima_weight, wei))
                    if not np.array_equal(ima_weight, wei):
                        print('wei:')
                        print(wei)
                        for ii in range(len(wei)):
                            print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                        print('ima_weight')
                        print(ima_weight)
                        for ii in range(len(ima_weight)):
                            print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))

                elif ('_all2a' in lay.name) or ('_all1' in lay.name):
                    ima_weight_new = []
                    for w in ima_weight:
                        if w.ndim == 1:
                            ima_weight_new.append(w)
                        elif w.ndim == 4:
                            ima_weight_new.append(np.concatenate((w, w), axis=2))

                    wei = np.array(lay.get_weights())
                    ima_weight_new = np.array(ima_weight_new)
                    print(lay.name)
                    print(np.array_equal(ima_weight_new, wei))
                    if not np.array_equal(ima_weight_new, wei):
                        print('wei:')
                        print(wei)
                        for ii in range(len(wei)):
                            print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                        print('ima_weight_new')
                        print(ima_weight_new)
                        for ii in range(len(ima_weight_new)):
                            print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight_new[ii].shape))

                else:
                    wei = np.array(lay.get_weights())
                    print(lay.name)
                    print(np.array_equal(ima_weight, wei))
                    if not np.array_equal(ima_weight, wei):
                        print('wei:')
                        print(wei)
                        for ii in range(len(wei)):
                            print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                        print('ima_weight')
                        print(ima_weight)
                        for ii in range(len(ima_weight)):
                            print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))
            else:
                wei = np.array(lay.get_weights())
                print(lay.name)
                print(np.array_equal(ima_weight, wei))
                if not np.array_equal(ima_weight, wei):
                    print('wei:')
                    print(wei)
                    for ii in range(len(wei)):
                        print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                    print('ima_weight')
                    print(ima_weight)
                    for ii in range(len(ima_weight)):
                        print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))

        elif '_geo' in lay.name:
            namecache = lay.name.replace('_geo', '')
            ima_weight = np.array(getWeightsForLayer(namecache, model_path))
            if 'res4' in lay.name:
                wei = np.array(lay.get_weights())
                print(lay.name)
                print(np.array_equal(ima_weight, wei))
                if not np.array_equal(ima_weight, wei):
                    print('wei:')
                    print(wei)
                    for ii in range(len(wei)):
                        print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                    print('ima_weight')
                    print(ima_weight)
                    for ii in range(len(ima_weight)):
                        print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))
            elif 'res5a' in lay.name:
                if ('geo2a' in lay.name) or ('_geo1' in lay.name):
                    ima_weight_new = []
                    for w in ima_weight:
                        if w.ndim == 1:
                            ima_weight_new.append(w)
                        elif w.ndim == 4:
                            ima_weight_new.append(np.concatenate((w, w), axis=2))
                    ima_weight_new = np.array(ima_weight_new)
                    wei = np.array(lay.get_weights())
                    print(lay.name)
                    print(np.array_equal(ima_weight_new, wei))
                    if not np.array_equal(ima_weight_new, wei):
                        print('wei:')
                        print(wei)
                        for ii in range(len(wei)):
                            print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                        print('ima_weight_new')
                        print(ima_weight_new)
                        for ii in range(len(ima_weight_new)):
                            print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight_new[ii].shape))

                else:
                    wei = np.array(lay.get_weights())
                    print(lay.name)
                    print(np.array_equal(ima_weight, wei))
                    if not np.array_equal(ima_weight, wei):
                        print('wei:')
                        print(wei)
                        for ii in range(len(wei)):
                            print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                        print('ima_weight')
                        print(ima_weight)
                        for ii in range(len(ima_weight)):
                            print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))
            else:
                wei = np.array(lay.get_weights())
                print(lay.name)
                print(np.array_equal(ima_weight, wei))
                if not np.array_equal(ima_weight, wei):
                    print('wei:')
                    print(wei)
                    for ii in range(len(wei)):
                        print('The shape of wei[{}]: {}'.format(ii, wei[ii].shape))
                    print('ima_weight')
                    print(ima_weight)
                    for ii in range(len(ima_weight)):
                        print('The shape of ima_weight[{}]: {}'.format(ii, ima_weight[ii].shape))
    print('The weight checking is finished!')


def freeze_layer_trainable(layer, exception_name):
    """
    Freeze the layer in training if its name doesnt contain specific names (this function must be embedded in the
    weight-setting function
    :param layer: the layer
    :param exception_name: set the names that dont need to be freezed in training
    :return: None
    """
    for name in exception_name:
        if name not in layer.name:
            layer.trainable = False



