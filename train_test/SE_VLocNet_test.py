from __future__ import division
import math
import helper
import resnet50
import SE_resnet50
import odo_data_preparation
import odometry_data
import numpy as np
from keras.optimizers import Adam, sgd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys

# The path of the weights

# 1.Path for original SE-VLocNet

checkpointname_se = 'checkpoint_weights_Loop_Pose.h5'
load_weight_name_se = 'custom_trained_weights_1_SGD_8.h5'
weight_path_se = './weights/KingsCollege/VLoc/se/'
load_weight_path_se = weight_path_se + load_weight_name_se
save_folder_path_se = './Test_Results/Seaborn_Plots/VLoc/SeResNet/origin/'  # 2/
save_csv_path_se = save_folder_path_se + 'origin SE-VLocNet Loop 50-150 Results.csv'

# 2.Path for v5

checkpointname_se_v5 = 'checkpoint_weights_Loop_Pose.h5'
load_weight_name_se_v5 = 'custom_trained_weights_1_SGD_8.h5'
weight_path_se_v5 = './weights/KingsCollege/VLoc/se/v5/'
load_weight_path_se_v5 = weight_path_se_v5 + load_weight_name_se_v5
save_folder_path_se_v5 = './Test_Results/Seaborn_Plots/VLoc/SeResNet/v5/'  # 2/
save_csv_path_se_v5 = save_folder_path_se_v5 + 'v5 SE-VLocNet Loop 50-150 Results.csv'

# 3.Path for transfer learning

checkpointname_geo_new = 'checkpoint_weights_VLoc_geo_new2019-03-14 10:49:26.h5'
load_weight_name_geo_new = 'custom_trained_weights_VLoc_geo2019-03-14 10:49:26.h5'
weight_path_geo_new = './weights/KingsCollege/VLoc/Geo/with_previous/'
load_weight_path_geo_new = weight_path_geo_new + load_weight_name_geo_new
save_folder_path_geo_new = './Test_Results/Seaborn_Plots/VLoc/ResNet/Geo/'  # 2/
save_csv_path_geo_new = save_folder_path_geo_new + 'VLocNet Geo with previous Position backfeeding Results.csv'

# 4.Path for v4

checkpointname_se_v4 = 'checkpoint_weights_VLoc_geo_se_v42019-03-15 10:57:41.h5'
load_weight_name_se_v4 = 'custom_trained_weights_VLoc_geo_se_v42019-03-15 10:57:41.h5'
weight_path_se_v4 = './weights/KingsCollege/VLoc/se/v4/'
load_weight_path_se_v4 = weight_path_se_v4 + load_weight_name_se_v4
save_folder_path_se_v4 = './Test_Results/Seaborn_Plots/VLoc/SeResNet/v4/'  # 2/
save_csv_path_se_v4 = save_folder_path_se_v4 + 'v4 SE-VLocNet Loop 50-150 Results.csv'

if __name__ == "__main__":

    dataset_odo_test = odometry_data.get_kings_VLoc_odo_test(odo_data_preparation.odo_save_test_path)

    # Shape the labels in form for next testing
    X_odo_test = np.squeeze(np.array(dataset_odo_test.images))
    X_odo_test_p = np.squeeze(np.array(dataset_odo_test.images_p))
    y_odo_test = np.squeeze(np.array(dataset_odo_test.odo_pose))

    y_odo_test_x = y_odo_test[:, 0:3]
    y_odo_test_q = y_odo_test[:, 3:7]

    # Load the model and set the optimizer
    mode = 4  # 1 for original SE-VlocNet with previous position feeding  2 for v5   3 for transfer learning   4 for v4

    # 1. odo net
    if mode == 1:
        print('Evaluation on original SE_VLocNet.')
        print('-----------------------------------')

        model = SE_resnet50.SE_VLocNet_v2(weights=None)

        print('Loading geometry data ........')

        dataset_geo_test = odometry_data.get_kings_VLoc_geo_new_test(odo_data_preparation.save_test)

        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        model.load_weights(load_weight_path_se)
        print('-----------------------------------')
        print('Loading odo model weight from: {}'.format(load_weight_path_se))
        print('-----------------------------------')
        print('/////Evaluation on progress//////')
        adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)
        model.compile(optimizer=adam, loss={'pose_merge': SE_resnet50.geo_loss,
                                            'odo_merge': SE_resnet50.odo_loss})
        test_predict = model.predict([X_odo_test_p, X_odo_test, x_geo_previous_test])
        print('Prediction: {}'.format(test_predict))
        print(test_predict[1][0])
        valsx_geo = test_predict[1][:, 7:10]
        valsq_geo = test_predict[1][:, 10:]
        results = np.zeros((len(dataset_geo_test.geo_pose), 2))
        for i in range(len(dataset_geo_test.geo_pose)):
            pose_geo_q = np.asarray(dataset_geo_test.geo_pose[i][3:7])
            pose_geo_x = np.asarray(dataset_geo_test.geo_pose[i][0:3])
            predicted_x = valsx_geo[i]
            predicted_q = valsq_geo[i]

            q1 = pose_geo_q / np.linalg.norm(pose_geo_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)

            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180/math.pi

            error_x = np.linalg.norm(pose_geo_x - predicted_x)

            results[i, :] = [error_x, theta]
            print('Iteration: ', i, ' Error XYZ (m): ', error_x, ' Error Q (degrees): ', theta)
        median_result = np.median(results, axis=0)
        print('Median error ', median_result[0], 'm and ', median_result[1], 'degrees.')

        if not math.isnan(median_result[0]):

            df = pd.DataFrame({'XYZ Error': [row[0] for row in results], 'Degree Error': [row[1] for row in results]})

            # Create a new plot area
            plt.figure()
            XYZ_label = 'Position Error Distribution (in m),(median error is:{0:.3f} m)'.format(median_result[0])
            q_label = 'Orientation Error Distribution (in degrees),(median error is:{0:.3f} degrees)'.format(
                median_result[1])

            Plot_XYZ = sns.distplot(df['XYZ Error'], kde= True, label=XYZ_label, bins=100)
            # Plot_XYZ.title('Distribution of XYZ Error(m)')
            # Plot_XYZ_save = Plot_XYZ.get_figure()
            # Plot_XYZ_save.savefig(save_XYZ_path)

            # sns.distplot(df['XYZ Error'], kde= True, label='Position Error Distribution (in m)')
            Plot_Q = sns.distplot(df['Degree Error'], kde=True, label=q_label, bins=100)
            plt.legend()
            Plot_Q_save = Plot_Q.get_figure()

            save_XYZ_path = save_folder_path_se + '_'+load_weight_name_se + '_XYZ.png'
            save_q_path = save_folder_path_se + '_'+load_weight_name_se + '_q.png'
            save_fig_path = save_folder_path_se + '_' +load_weight_name_se + '_Dist.png'
            Plot_Q_save.savefig(save_fig_path)
            print('The plot is created at: ' + save_fig_path)

        else:
            sys.stdout = open(save_folder_path_se + '_' + load_weight_name_se + '_Result.txt', 'w')
            print('The .txt file is created at: ' + save_folder_path_se + '_' + load_weight_name_se + '_Result.txt')

    elif mode == 2:
        # 2. v5
        print('Evaluation on v5 SE_VLocNet.')
        print('-----------------------------------')

        model = SE_resnet50.SE_VLocNet_v5(weights=None)

        print('Loading geometry data ........')

        dataset_geo_test = odometry_data.get_kings_VLoc_geo_new_test(odo_data_preparation.save_test)

        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        model.load_weights(load_weight_path_se_v5)
        print('-----------------------------------')
        print('Loading v5 model weight from: {}'.format(load_weight_path_se_v5))
        print('-----------------------------------')
        print('/////Evaluation on progress//////')
        adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)
        model.compile(optimizer=adam, loss={'pose_merge': SE_resnet50.geo_loss,
                                            'odo_merge': SE_resnet50.odo_loss})
        test_predict = model.predict([X_odo_test_p, X_odo_test, x_geo_previous_test])
        print('Prediction: {}'.format(test_predict))
        print(test_predict[1][0])
        valsx_geo = test_predict[1][:, 7:10]
        valsq_geo = test_predict[1][:, 10:]
        results = np.zeros((len(dataset_odo_test.images), 2))
        for i in range(len(dataset_geo_test.geo_pose)):
            pose_geo_q = np.asarray(dataset_geo_test.geo_pose[i][3:7])
            pose_geo_x = np.asarray(dataset_geo_test.geo_pose[i][0:3])
            predicted_x = valsx_geo[i]
            predicted_q = valsq_geo[i]

            q1 = pose_geo_q / np.linalg.norm(pose_geo_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)

            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180/math.pi

            error_x = np.linalg.norm(pose_geo_x - predicted_x)

            results[i, :] = [error_x, theta]
            print('Iteration: ', i, ' Error XYZ (m): ', error_x, ' Error Q (degrees): ', theta)
        median_result = np.median(results, axis=0)
        print('Median error ', median_result[0], 'm and ', median_result[1], 'degrees.')
        if not math.isnan(median_result[0]):

            df = pd.DataFrame({'XYZ Error': [row[0] for row in results], 'Degree Error':[row[1] for row in results]})

            # Create a new plot area
            plt.figure()
            XYZ_label = 'Position Error Distribution (in m),(median error is:{0:.3f} m)'.format(median_result[0])
            q_label = 'Orientation Error Distribution (in degrees),(median error is:{0:.3f} degrees)'.format(
                median_result[1])

            Plot_XYZ = sns.distplot(df['XYZ Error'], kde= True, label=XYZ_label, bins=100)
            # Plot_XYZ.title('Distribution of XYZ Error(m)')
            # Plot_XYZ_save = Plot_XYZ.get_figure()
            # Plot_XYZ_save.savefig(save_XYZ_path)

            # sns.distplot(df['XYZ Error'], kde= True, label='Position Error Distribution (in m)')
            Plot_Q = sns.distplot(df['Degree Error'], kde=True, label=q_label, bins=100)
            plt.legend()
            Plot_Q_save = Plot_Q.get_figure()

            save_XYZ_path = save_folder_path_se_v5 + '_'+load_weight_name_se_v5 + '_XYZ.png'
            save_q_path = save_folder_path_se_v5 + '_'+load_weight_name_se_v5 + '_q.png'
            save_fig_path = save_folder_path_se_v5 + '_' + load_weight_name_se_v5 + '_Dist.png'
            Plot_Q_save.savefig(save_fig_path)
            print('The plot is created at: ' + save_fig_path)

        else:
            sys.stdout = open(save_folder_path_se_v5 + '_' + load_weight_name_se_v5 + '_Result.txt', 'w')
            print('The .txt file is created at: ' + save_folder_path_se_v5 + '_' + load_weight_name_se_v5 +
                  '_Result.txt')

    elif mode == 3:
        # 3. SE-VLocNet for transfer learning  ## under construction!!!
        print('Evaluation on original SE_VLocNet on transfer learning principal.')
        print('-----------------------------------')
        model = resnet50.VLocNet_v2(weights=None)

        print('Loading geometry data ........')

        dataset_geo_test = odometry_data.get_kings_VLoc_geo_new_test(odo_data_preparation.save_test)

        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        model.load_weights(load_weight_path_geo_new)
        print('-----------------------------------')
        print('Loading odo model weight from: {}'.format(load_weight_path_geo_new))
        print('-----------------------------------')
        print('/////Evaluation on progress//////')
        adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)
        model.compile(optimizer=adam, loss={'pose_merge': resnet50.geo_loss,
                                            'odo_merge': resnet50.odo_loss})
        test_predict = model.predict([X_odo_test_p, X_odo_test, x_geo_previous_test])
        print('Prediction: {}'.format(test_predict))
        print(test_predict[1][0])
        valsx_geo = test_predict[1][:, 7:10]
        valsq_geo = test_predict[1][:, 10:]
        results = np.zeros((len(dataset_geo_test.geo_pose), 2))
        for i in range(len(dataset_geo_test.geo_pose)):
            pose_geo_q = np.asarray(dataset_geo_test.geo_pose[i][3:7])
            pose_geo_x = np.asarray(dataset_geo_test.geo_pose[i][0:3])
            predicted_x = valsx_geo[i]
            predicted_q = valsq_geo[i]
            q1 = pose_geo_q / np.linalg.norm(pose_geo_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)

            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180/math.pi

            error_x = np.linalg.norm(pose_geo_x - predicted_x)

            results[i, :] = [error_x, theta]
            print('Iteration: ', i, ' Error XYZ (m): ', error_x, ' Error Q (degrees): ', theta)
        median_result = np.median(results, axis=0)
        print('Median error ', median_result[0], 'm and ', median_result[1], 'degrees.')

        if not math.isnan(median_result[0]):

            df = pd.DataFrame({'XYZ Error': [row[0] for row in results], 'Degree Error': [row[1] for row in results]})

            # Create a new plot area
            plt.figure()
            XYZ_label = 'Position Error Distribution (in m),(median error is:{0:.3f} m)'.format(median_result[0])
            q_label = 'Orientation Error Distribution (in degrees),(median error is:{0:.3f} degrees)'.format(
                median_result[1])

            Plot_XYZ = sns.distplot(df['XYZ Error'], kde= True, label=XYZ_label, bins=100)
            # Plot_XYZ.title('Distribution of XYZ Error(m)')
            # Plot_XYZ_save = Plot_XYZ.get_figure()
            # Plot_XYZ_save.savefig(save_XYZ_path)

            # sns.distplot(df['XYZ Error'], kde= True, label='Position Error Distribution (in m)')
            Plot_Q = sns.distplot(df['Degree Error'], kde=True, label=q_label, bins=100)
            plt.legend()
            Plot_Q_save = Plot_Q.get_figure()

            save_XYZ_path = save_folder_path_geo_new + '_'+load_weight_name_geo_new + '_XYZ.png'
            save_q_path = save_folder_path_geo_new + '_'+load_weight_name_geo_new + '_q.png'
            save_fig_path = save_folder_path_geo_new + '_' + load_weight_name_geo_new + '_Dist.png'
            Plot_Q_save.savefig(save_fig_path)
            print('The plot is created at: ' + save_fig_path)

        else:
            sys.stdout = open(save_folder_path_geo_new + '_' + load_weight_name_geo_new + '_Result.txt', 'w')
            print('The .txt file is created at: ' + save_folder_path_geo_new + '_' + load_weight_name_geo_new
                  + '_Result.txt')

    elif mode == 4:
        # 4. v4 net
        print('Evaluation on v4 SE_VLocNet.')
        print('-----------------------------------')

        model = SE_resnet50.SE_VLocNet_v4(weights=None)

        dataset_geo_test = odometry_data.get_kings_VLoc_geo_new_test(odo_data_preparation.save_test)

        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        model.load_weights(load_weight_path_se_v4)
        print('-----------------------------------')
        print('Loading odo model weight from: {}'.format(load_weight_path_se_v4))
        print('-----------------------------------')
        print('/////Evaluation on progress//////')
        adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)
        model.compile(optimizer=adam, loss={'pose_merge': SE_resnet50.geo_loss,
                                            'odo_merge': SE_resnet50.odo_loss})
        test_predict = model.predict([X_odo_test_p, X_odo_test])
        print(test_predict)
        valsx_geo = np.array([row[7:10] for row in test_predict[1]])
        valsq_geo = np.array([row[10:] for row in test_predict[1]])
        print('the shape of valsx_geo is: {}'.format(valsx_geo.shape))
        print('the shape of valsq_geo is: {}'.format(valsq_geo.shape))
        results = np.zeros((len(dataset_geo_test.geo_pose), 2))

        for i in range(len(dataset_geo_test.geo_pose)):
            pose_geo_q = np.squeeze(np.asarray(dataset_geo_test.geo_pose[i][3:7]))
            pose_geo_x = np.squeeze(np.asarray(dataset_geo_test.geo_pose[i][0:3]))
            predicted_x = valsx_geo[i]
            predicted_q = valsq_geo[i]
            print('the shape of predicted_q is: {}'.format(predicted_q.shape))
            print('the shape of valsq_geo is: {}'.format(valsq_geo.shape))

            q1 = pose_geo_q/np.linalg.norm(pose_geo_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180/math.pi

            error_x = np.linalg.norm(pose_geo_x - predicted_x)

            results[i, :] = [error_x, theta]
            print('Iteration: ', i, ' Error XYZ (m): ', error_x, ' Error Q (degrees): ', theta)
        median_result = np.median(results, axis=0)
        print('Median error ', median_result[0], 'm and ', median_result[1], 'degrees.')

        if not math.isnan(median_result[0]):

            df = pd.DataFrame({'XYZ Error': [row[0] for row in results], 'Degree Error': [row[1] for row in results]})

            # Create a new plot area
            plt.figure()
            XYZ_label = 'Position Error Distribution (in m),(median error is:{0:.3f} m)'.format(median_result[0])
            q_label = 'Orientation Error Distribution (in degrees),(median error is:{0:.3f} degrees)'.format(
                median_result[1])

            Plot_XYZ = sns.distplot(df['XYZ Error'], kde= True, label=XYZ_label, bins=100)
            # Plot_XYZ.title('Distribution of XYZ Error(m)')
            # Plot_XYZ_save = Plot_XYZ.get_figure()
            # Plot_XYZ_save.savefig(save_XYZ_path)

            # sns.distplot(df['XYZ Error'], kde= True, label='Position Error Distribution (in m)')
            Plot_Q = sns.distplot(df['Degree Error'], kde=True, label=q_label, bins=100)
            plt.legend()
            Plot_Q_save = Plot_Q.get_figure()

            save_XYZ_path = save_folder_path_se_v4 + '_'+load_weight_name_se_v4 + '_XYZ.png'
            save_q_path = save_folder_path_se_v4 + '_'+load_weight_name_se_v4 + '_q.png'
            save_fig_path = save_folder_path_se_v4 + '_' + load_weight_name_se_v4 + '_Dist.png'
            Plot_Q_save.savefig(save_fig_path)
            print('The plot is created at: ' + save_fig_path)

        else:
            sys.stdout = open(save_folder_path_se_v4 + '_' + load_weight_name_se_v4 + '_Result.txt', 'w')
            print('The .txt file is created at: ' + save_folder_path_se_v4 + '_' + load_weight_name_se_v4
                  + '_Result.txt')
    print('Test is finished.')
