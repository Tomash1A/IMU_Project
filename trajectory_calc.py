from Utils import *
import numpy as np
from docutils.nodes import inline
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ahrs, matplotlib

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# matplotlib.use('Qt5Agg')  # Use TkAgg backend to display plots in a new window
plt.close('all')

# data acquisition + Parameter callings
data_path = r"C:\Users\user1\PycharmProjects\SNN-SCTN\TomerNir2024\indoor_forward_9_davis_with_gt.zip"
gt_path = r"C:\Users\user1\University_ 4th year\project\C mahony\pythonProject\.venv\groundtruth.txt"
walk_csv_path = r'C:\Users\user1\University_ 4th year\project\C mahony\pythonProject\.venv\short_walk.csv'


# # # For CSV dataset- warning: works with g and deg/s
# data = read_imu_data_from_csv(walk_csv_path)      #for long/short walk CSV dataset
# calib_start, calib_end  = 0, 600  #  for CSV
# start, stop = 0, -1


# # # Parameters:
calib_start, calib_end  = 0, 3000  #  calib_start, calib end = 23033, 30000 for dataset3
start, stop = 30000, 80000
cutoff = 25
fs = 1000       # for UZH dataset- samples every ~1 [ms]
LPF_order = 2

# Load data
data = read_imu_data(data_path)


# # optional- filter the data before breaking it down to arrayes
# data = apply_lpf(data, cutoff, fs, order=LPF_order)

# Get Calibration data part from whole IMU data, plot the Magnitude and samples:
cal_timestamp, cal_gyroscope, cal_accelerometer = breakdown_data(data, calib_start, calib_end, fix_axis_and_g=1,no_Z_gyro=0, no_Z_accel=0,fix_meas=0,g=9.8065)
cal_err_lin, cal_err_lin_world, cal_err_gyr = get_imu_calibration(cal_timestamp, cal_accelerometer,cal_gyroscope)
fig = plot_raw_data(cal_timestamp, cal_gyroscope, cal_accelerometer)
plt.show()

# Get Flying data part (without calibration) from whole IMU data, plot the Magnitude and samples:
timestamp, gyroscope, accelerometer = breakdown_data(data, start, stop, fix_axis_and_g=1,no_Z_gyro=0, no_Z_accel=0,fix_meas=0,g=9.8065)
gyroscope = gyroscope - cal_err_gyr             #compensate for the noise measured in calibration segment
# accelerometer = accelerometer -cal_err_lin      #compensate for the noise measured in calibration segment do not double compnsate in calc_position_from_IMU
fig = plot_raw_data(timestamp, gyroscope, accelerometer)
plt.show()



def calc_position_from_IMU(timestamp, gyroscope, accelerometer, cal_err_lin, cal_err_lin_world, cal_err_gyr,decrease_world_frame_acc_err=0):
    print("##-------Data parameters--------------------", '\n')
    # gyroscope = gyroscope - cal_err_gyr             #compensate for the noise measured in calibration segment
    # accelerometer = accelerometer -cal_err_lin      #compensate for the noise measured in calibration segment
    orientation = ahrs.filters.Mahony()
    N = len(gyroscope)
    Q = np.tile([1., 0., 0., 0.], (N, 1)) #create a dataset for Q
    points = np.zeros(shape=(N, 3))
    velocity_world = np.zeros(shape=(3,))
    diff_accel_norm = 0
    Q_norm = 0
    for i in range(N):
        dt = timestamp[i]
        orientation.Dt = dt
        Q[i] = orientation.updateIMU(Q[i - 1], gyr=gyroscope[i], acc=accelerometer[i])  #Create new Quat from old one + sensor readings
        R = Rotation.from_quat(Q[i]).as_matrix()
        accel_body = accelerometer[i]
        if decrease_world_frame_acc_err:
            accel_world = (R @ accel_body) - cal_err_lin_world  # compansate for error in world orientation
        else:
            accel_world = (R @ accel_body) - cal_err_lin_world        #compansate for error in world orientation
        # accel_world = (R @ accel_body)
        velocity_world = velocity_world + accel_world * dt
        points[i] = points[i - 1] + velocity_world * dt
        diff_accel_norm = np.abs(np.linalg.norm(accel_body) - np.linalg.norm(accel_world))
        Q_norm += np.linalg.norm(Q[i])              # For debugging- ||Q||_avg should be 1,
    print('||a_world - a_body||_avg= ', diff_accel_norm / N, '\n', '||Q||_avg = ', Q_norm / N, '\n')
    calculate_and_plot_from_Q(points, Q)
    return points, Q

points, quats = calc_position_from_IMU(timestamp, gyroscope, accelerometer, cal_err_lin, cal_err_lin_world, cal_err_gyr,decrease_world_frame_acc_err=1)

# cal_gyroscope[:,:], cal_accelerometer[:,:]=0,0    #unhash to see the error resonating
cal_points, cal_quats = calc_position_from_IMU(cal_timestamp,cal_gyroscope, cal_accelerometer,0,0,0,decrease_world_frame_acc_err=1)

# Calaculate and plot for Ground Truth
# Data extraction from Ground Truth file.
gt_t_xyz, gt_q_wxyz, gt_data = get_gt_data(gt_path)
gt_start, gt_stop = 675, len(gt_data)//4       # (//4) for one loop only
calculate_and_plot_from_Q(gt_t_xyz[gt_start:gt_stop], gt_q_wxyz[gt_start:gt_stop],)


# position, velocity, rotation_matrix = dead_reckoning_3d(timestamp, accelerometer, gyroscope)
# plot_trajectory_3d(position, rotation_matrix)





















