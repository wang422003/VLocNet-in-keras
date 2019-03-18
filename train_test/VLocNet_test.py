from __future__ import division
import math
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

# 1.Path for odo net

checkpointname_odo = 'checkpoint_weights_Loop_Pose.h5'
load_weight_name_odo = 'custom_trained_weights_1_SGD_8.h5'
weight_path_odo = './weights/KingsCollege/'
load_weight_path_odo = weight_path_odo + load_weight_name_odo
save_folder_path_odo = './Test_Results/Seaborn_Plots/PoseNet/modi/'  # 2/
save_csv_path_odo = save_folder_path_odo + 'modified PoseNet(no icp9) Loop with LRN 50-150 Results.csv'

# 2.Path for geo net without previous position backfeeding

checkpointname_geo = 'checkpoint_weights_Loop_Pose.h5'
load_weight_name_geo = 'custom_trained_weights_1_SGD_8.h5'
weight_path_geo = './weights/KingsCollege/'
load_weight_path_geo = weight_path_geo + load_weight_name_geo
save_folder_path_geo = './Test_Results/Seaborn_Plots/PoseNet/modi/'  # 2/
save_csv_path_geo = save_folder_path_geo + 'modified PoseNet(no icp9) Loop with LRN 50-150 Results.csv'

# 3.Path for geo net with previous position backfeeding

checkpointname_geo_new = 'custom_trained_weights_VLoc_geo_IN2019-03-15 11:14:40.h5'
load_weight_name_geo_new = 'custom_trained_weights_VLoc_geo_IN2019-03-15 11:14:40.h5'
weight_path_geo_new = './weights/KingsCollege/VLoc/Geo/with_previous/'
load_weight_path_geo_new = weight_path_geo_new + load_weight_name_geo_new
save_folder_path_geo_new = './Test_Results/Seaborn_Plots/VLoc/ResNet/Geo/'  # 2/
save_csv_path_geo_new = save_folder_path_geo_new + 'VLocNet Geo with previous Position backfeeding Results.csv'

# 4.Path for se net with previous position backfeeding

checkpointname_geo_se = 'checkpoint_weights_Loop_Pose.h5'
load_weight_name_geo_se = 'custom_trained_weights_1_SGD_8.h5'
weight_path_geo_se = './weights/KingsCollege/'
load_weight_path_geo_se = weight_path_geo_se + load_weight_name_geo_se
save_folder_path_geo_se= './Test_Results/Seaborn_Plots/PoseNet/modi/'  # 2/
save_csv_path_geo_nse = save_folder_path_geo_se + 'modified PoseNet(no icp9) Loop with LRN 50-150 Results.csv'
# save_XYZ_path = './Test_Results/Seaborn_Plots/xx_XYZ.png'
# save_q_path = './Test_Results/Seaborn_Plots/xx_q.png'
# save_fig_path = './Test_Results/Seaborn_Plots/xx_Dist.png'

if __name__ == "__main__":

    dataset_odo_test = odometry_data.get_kings_VLoc_odo_test(odo_data_preparation.odo_save_test_path)

    # Shape the labels in form for next testing
    X_odo_test = np.squeeze(np.array(dataset_odo_test.images))
    X_odo_test_p = np.squeeze(np.array(dataset_odo_test.images_p))
    y_odo_test = np.squeeze(np.array(dataset_odo_test.odo_pose))

    y_odo_test_x = y_odo_test[:, 0:3]
    y_odo_test_q = y_odo_test[:, 3:7]

    # Load the model and set the optimizer
    mode = 3  # 1 for odo, 2 for geo without previous position backfeeding, 3 for geo with previous position backfeeding
              # 4 for se geo net
    # 1. odo net
    if mode == 1:
        model = resnet50.VLocNet_Odometry_new()
        model.load_weights(load_weight_path_odo)
        print('-----------------------------------')
        print('Loading odo model weight from: {}'.format(load_weight_path_odo))
        print('-----------------------------------')
        print('/////Evaluation on progress//////')
        adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)
        model.compile(optimizer=adam, loss={'fc_2': resnet50.euc_lossx_odo,
                                            'fc_3': resnet50.euc_lossq_odo})
        test_predict = model.predict([X_odo_test_p, X_odo_test])
        valsx_odo = test_predict[0]
        valsq_odo = test_predict[1]
        results = np.zeros((len(dataset_odo_test.images), 2))
        for i in range(len(dataset_odo_test.images)):
            pose_odo_q = np.squeeze(np.asarray(dataset_odo_test.odo_pose[i][3:7]))
            pose_odo_x = np.squeeze(np.asarray(dataset_odo_test.odo_pose[i][0:3]))
            predicted_x = valsx_odo[i]
            predicted_q = valsq_odo[i]

            q1 = pose_odo_q/np.linalg.norm(pose_odo_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)
            d = abs(np.sum(np.multiply(q1, q2)))
            theta = 2 * np.arccos(d) * 180/math.pi

            error_x = np.linalg.norm(pose_odo_x - predicted_x)

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

            save_XYZ_path = save_folder_path_odo + '_'+load_weight_name_odo + '_XYZ.png'
            save_q_path = save_folder_path_odo + '_'+load_weight_name_odo + '_q.png'
            save_fig_path = save_folder_path_odo + '_' +load_weight_name_odo + '_Dist.png'
            Plot_Q_save.savefig(save_fig_path)
            print('The plot is created at: ' + save_fig_path)

        else:
            sys.stdout = open(save_folder_path_odo + '_' + load_weight_name_odo + '_Result.txt', 'w')
            print('The .txt file is created at: ' +save_folder_path_odo + '_' + load_weight_name_odo + '_Result.txt')

    elif mode == 2:
        # 2. geo net without backfeeded previous position
        print('Loading geometry data ........')

        dataset_geo_train, dataset_geo_test = odometry_data.get_kings_VLoc_geo(odo_data_preparation.save_train,
                                                                               odo_data_preparation.save_test)

        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))

        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        model = resnet50.VLocNet_full()
        model.load_weights(load_weight_path_geo)
        print('-----------------------------------')
        print('Loading odo model weight from: {}'.format(load_weight_path_geo))
        print('-----------------------------------')
        print('/////Evaluation on progress//////')
        adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)
        model.compile(optimizer=adam, loss={'pose_merge': resnet50.geo_loss,
                                            'odo_merge': resnet50.odo_loss})
        test_predict = model.predict([X_odo_test_p, X_odo_test])
        valsx_geo = test_predict[1][7:10]
        valsq_geo = test_predict[1][10:]
        results = np.zeros((len(dataset_geo_test.images), 2))
        for i in range(len(dataset_geo_test.images)):
            pose_geo_q = np.squeeze(np.asarray(dataset_geo_test.geo_pose[i][3:7]))
            pose_geo_x = np.squeeze(np.asarray(dataset_geo_test.geo_pose[i][0:3]))
            predicted_x = valsx_geo[i]
            predicted_q = valsq_geo[i]

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

            save_XYZ_path = save_folder_path_geo + '_'+load_weight_name_geo + '_XYZ.png'
            save_q_path = save_folder_path_geo + '_'+load_weight_name_geo + '_q.png'
            save_fig_path = save_folder_path_geo + '_' + load_weight_name_geo + '_Dist.png'
            Plot_Q_save.savefig(save_fig_path)
            print('The plot is created at: ' + save_fig_path)

        else:
            sys.stdout = open(save_folder_path_geo + '_' + load_weight_name_geo + '_Result.txt', 'w')
            print('The .txt file is created at: ' + save_folder_path_geo + '_' + load_weight_name_geo + '_Result.txt')

    elif mode == 3:
        # 3. geo net with backfeeded previous position
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
            # print('type of pose_geo_q is:{}'.format(type(pose_geo_q)))
            # print('pose_geo_q : {}'.format(pose_geo_q))
            # print('type of the element in pose_geo_q: {}'.format(type(pose_geo_q[0])))
            # print('type of norm is: {}'.format(type(np.linalg.norm(pose_geo_q))))
            # print('norm: {}'.format(np.linalg.norm(pose_geo_q)))
            q1 = pose_geo_q / np.linalg.norm(pose_geo_q)
            q2 = predicted_q / np.linalg.norm(predicted_q)

            # q1 = np.divide(pose_geo_q, np.linalg.norm(pose_geo_q))
            # q2 = np.divide(predicted_q, np.linalg.norm(predicted_q))
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
        # 3. geo net with backfeeded previous position
        model = SE_resnet50.SE_VLocNet_v2()

        print('Loading geometry data ........')

        dataset_geo_test = odometry_data.get_kings_VLoc_geo_new_test(odo_data_preparation.save_test)

        y_geo_test = np.squeeze(np.array(dataset_geo_test.geo_pose))
        x_geo_previous_test = np.squeeze(np.array(dataset_geo_test.geo_pose_previous))

        merge_test = np.concatenate((y_odo_test, y_geo_test), axis=1)

        model.load_weights(load_weight_path_geo_se)
        print('-----------------------------------')
        print('Loading odo model weight from: {}'.format(load_weight_path_geo_se))
        print('-----------------------------------')
        print('/////Evaluation on progress//////')
        adam = Adam(beta_1=0.9, beta_2=0.999, lr=0.0001, epsilon=0.0000000001)
        model.compile(optimizer=adam, loss={'pose_merge': SE_resnet50.geo_loss,
                                            'odo_merge': SE_resnet50.odo_loss})
        test_predict = model.predict([X_odo_test_p, X_odo_test, x_geo_previous_test])
        valsx_geo = np.array([row[7:10] for row in test_predict[1]])
        valsq_geo = np.array([row[10:] for row in test_predict[1]])
        results = np.zeros((len(dataset_geo_test.images), 2))
        for i in range(len(dataset_geo_test.images)):
            pose_geo_q = np.squeeze(np.asarray(dataset_geo_test.geo_pose[i][3:7]))
            pose_geo_x = np.squeeze(np.asarray(dataset_geo_test.geo_pose[i][0:3]))
            predicted_x = valsx_geo[i]
            predicted_q = valsq_geo[i]

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

            save_XYZ_path = save_folder_path_geo_se + '_'+load_weight_name_geo_se + '_XYZ.png'
            save_q_path = save_folder_path_geo_se + '_'+load_weight_name_geo_se + '_q.png'
            save_fig_path = save_folder_path_geo_se + '_' + load_weight_name_geo_se + '_Dist.png'
            Plot_Q_save.savefig(save_fig_path)
            print('The plot is created at: ' + save_fig_path)

        else:
            sys.stdout = open(save_folder_path_geo_se + '_' + load_weight_name_geo_se + '_Result.txt', 'w')
            print('The .txt file is created at: ' + save_folder_path_geo_se + '_' + load_weight_name_geo_se
                  + '_Result.txt')
    print('Test is finished.')
