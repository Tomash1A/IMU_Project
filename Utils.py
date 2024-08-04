############ utility Functions for the project #########################
import zipfile
# import numpy
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import ahrs, matplotlib
import numpy as np
from scipy import signal
from matplotlib.ticker import MaxNLocator
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def read_imu_data(zip_path):
    serial_num = []
    timestamps = []
    wx_values = []
    wy_values = []
    wz_values = []
    ax_values = []
    ay_values = []
    az_values = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('imu.txt') as imu_file:
            next(imu_file)  # Skip the first line
            for line in imu_file:
                line = line.decode('utf-8').strip().split()
                sn, ts, wx, wy, wz, ax, ay, az = map(float, line)
                serial_num.append(sn)
                timestamps.append(ts)
                wx_values.append(wx)
                wy_values.append(wy)
                wz_values.append(wz)
                ax_values.append(ax)
                ay_values.append(ay)
                az_values.append(az)

    imu_data = np.zeros((len(ax_values),7))
    imu_data[:, 0] = timestamps
    imu_data[:, 1] = wx_values
    imu_data[:, 2] = wy_values
    imu_data[:, 3] = wz_values
    imu_data[:, 4] = ax_values
    imu_data[:, 5] = ay_values
    imu_data[:, 6] = az_values
    # print(imu_data)
    return imu_data

def read_imu_data_from_csv(csv_path):

    timestamps = []
    wx_values = []
    wy_values = []
    wz_values = []
    ax_values = []
    ay_values = []
    az_values = []

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Iterate through each row in the dataframe
    for index, row in df.iterrows():
        ts, wx, wy, wz, ax, ay, az = row
        timestamps.append(ts)
        wx_values.append(wx)
        wy_values.append(wy)
        wz_values.append(wz)
        ax_values.append(ax)
        ay_values.append(ay)
        az_values.append(az)

    imu_data = np.zeros((len(ax_values), 7))
    imu_data[:, 0] = timestamps
    imu_data[:, 1] = wx_values
    imu_data[:, 2] = wy_values
    imu_data[:, 3] = wz_values
    imu_data[:, 4] = ax_values
    imu_data[:, 5] = ay_values
    imu_data[:, 6] = az_values

    return imu_data

def plot_positions(position):
    positions = position
    # Plot the 3D trace
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Space Trace from Quaternions')
    plt.show()

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def plot_raw_data(cal_timestamp, cal_gyroscope, cal_accelerometer):
    # Calculate vector magnitudes
    gyro_mag = np.linalg.norm(cal_gyroscope, axis=1)
    accel_mag = np.linalg.norm(cal_accelerometer, axis=1)

    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('IMU Raw Data')

    # Plot 1: Gyroscope magnitude
    axs[0, 0].plot(cal_timestamp, gyro_mag)
    axs[0, 0].set_title('Gyroscope Magnitude')
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Magnitude')

    # Plot 2: Accelerometer magnitude
    axs[0, 1].plot(cal_timestamp, accel_mag)
    axs[0, 1].set_title('Accelerometer Magnitude')
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Magnitude')

    # Plot 3: Individual gyroscope components
    axs[1, 0].plot(cal_timestamp, cal_gyroscope[:, 0], label='gx')
    axs[1, 0].plot(cal_timestamp, cal_gyroscope[:, 1], label='gy')
    axs[1, 0].plot(cal_timestamp, cal_gyroscope[:, 2], label='gz')
    axs[1, 0].set_title('Gyroscope Components')
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Angular Velocity')
    axs[1, 0].legend()

    # Plot 4: Individual accelerometer components
    axs[1, 1].plot(cal_timestamp, cal_accelerometer[:, 0], label='ax')
    axs[1, 1].plot(cal_timestamp, cal_accelerometer[:, 1], label='ay')
    axs[1, 1].plot(cal_timestamp, cal_accelerometer[:, 2], label='az')
    axs[1, 1].set_title('Accelerometer Components')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Acceleration')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()

def breakdown_data(data, start, stop,fix_axis_and_g=0, no_Z_gyro=0, no_Z_accel= 0, fix_meas=0,g=9.8065):
    ts = data[start:stop, 0]
    # UZH Data set default: rad/s, use np.degrees(  ) in order to convert to 1/s
    gyro = data[start:stop,1:4]
    # UZH Data set default: m/(s^2), use ()/(9.8065) in order to convert to g
    accel = data[start:stop,4:7]
    ts = ts - ts[0]
    if fix_meas:
        accel, gyro = fix_measurements_and_g(accel, gyro)   #for datasets in deg/s and g + decrease g
    if fix_axis_and_g:
        accel, gyro = fix_axis_an_g(accel, gyro, g)
    if no_Z_gyro:
        gyro[:,2] = 0
    if no_Z_accel:
        accel[:, 2] = 0
    return ts, gyro, accel

def calculate_and_plot_from_Q(gt_t_xyz, gt_q_wxyz):
    tx, ty, tz = gt_t_xyz[:,0], gt_t_xyz[:,1], gt_t_xyz[:,2]
    qw, qx, qy, qz, = gt_q_wxyz[:,0], gt_q_wxyz[:,1], gt_q_wxyz[:,2], gt_q_wxyz[:,3]
    # Convert quaternions to rotation objects
    rotations = Rotation.from_quat(np.column_stack((qx, qy, qz, qw)))

    # Create a simple drone model
    drone_points = np.array([
        [0.5, 0, 0],  # Nose
        [-0.5, 0.5, 0],  # Left wing
        [-0.5, -0.5, 0],  # Right wing
        [-0.5, 0, 0.2]  # Tail
    ])

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot trajectory
    ax.plot(tx, ty, tz, 'b-', linewidth=2, label='Trajectory')

    # Plot drone orientation at intervals
    N = (gt_t_xyz.shape[0])
    interval = (gt_t_xyz.shape[0]) // 20  # Adjust this to change the number of orientations shown
    for i in range(0, N, interval):
        # Apply rotation and translation for this specific time step
        rotated_points = rotations[i].apply(drone_points)
        translated_points = rotated_points + np.array([tx[i], ty[i], tz[i]])

        # Plot the drone
        ax.plot(translated_points[:, 0], translated_points[:, 1], translated_points[:, 2], 'r-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Drone Trajectory and Orientation')

    plt.tight_layout()
    plt.show()

def get_imu_calibration(timestamp, accel_data, gyro_data):
    print("##-------Calibration parameters--------------------", '\n')
    # Calculate mean errors
    orientation = ahrs.filters.Mahony()
    N = len(gyro_data)
    Q = np.tile([1., 0., 0., 0.], (N, 1))  # create an array for Q
    accel_world = np.zeros(shape=(3,))
    diff_accel_norm = 0
    for i in range(N):
        dt = timestamp[i]
        orientation.Dt = dt  # set net dt as  t(i)-t(i-1)
        Q[i] = orientation.updateIMU(Q[i - 1], gyr=gyro_data[i], acc=accel_data[i])  # use Mahony in order to get Q
        R = Rotation.from_quat(Q[i]).as_matrix()    #use Q in order to get R matrix
        accel_body = accel_data[i]
        accel_world = (R @ accel_body.reshape(3, 1)).reshape(3, )
        diff_accel_norm += (np.linalg.norm(accel_body)- np.linalg.norm(accel_world))
        # print("difference norm: ", (np.linalg.norm(accel_body)- np.linalg.norm(accel_world)))
    cal_err_lin_world = np.mean(gyro_data, axis=0)
    print("Diffrences W and B accel avg:", '\n', diff_accel_norm/N, '\n')   #use for Sanity check
    # For accelerometer, subtract expected gravity (assuming Z-axis is vertical)
    # Adjust this if your IMU has a different orientation
    # cal_err_lin[2] -= 9.81  # Subtract 1g from z-axis
    cal_err_lin = np.mean(accel_data, axis=0)
    cal_err_gyr = np.mean(gyro_data, axis=0)
    print("Linear acceleartion error-Body:", '\n', cal_err_lin)
    print("Linear acceleartion error-World:", '\n', cal_err_lin_world)
    print("Gyroscope error:", '\n', cal_err_gyr, '\n')
    print("Accelration Error Mean in World Frame: ", np.linalg.norm(cal_err_lin_world), '\n')
    return cal_err_lin, cal_err_lin_world, cal_err_gyr

def get_gt_data(gt_path):
    gt_data = np.loadtxt(gt_path, skiprows=1)
    gt_t_xyz = gt_data[:, 1:4]
    gt_q_wxyz = gt_data[:, 4:]
    return gt_t_xyz, gt_q_wxyz , gt_data

def fix_axis_an_g(accel,gyro, g):
    gravity_vec = np.array([0, 0, -g])
    R_y_to_z = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    # fix axis
    return (np.dot(accel, R_y_to_z.T) -gravity_vec), (np.dot(gyro, R_y_to_z.T))

def fix_measurements_and_g(accel,gyro):
    accel = (accel * 9.8065)
    accel[:,2] -= 9.8065
    return accel, (gyro* (np.pi/180))

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lpf(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    data = data.transpose()
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered_data[i] = signal.filtfilt(b, a, data[i])
    return filtered_data.transpose()

def calc_position_from_IMU_live(timestamp, gyroscope, accelerometer, cal_err_lin, cal_err_lin_world, cal_err_gyr):
    gyroscope = gyroscope - cal_err_gyr
    accelerometer = accelerometer
    orientation = ahrs.filters.Mahony()
    N = len(gyroscope)
    Q = np.tile(np.array([1., 0., 0., 0.]), (N, 1))
    points = np.zeros(shape=(N, 3))
    velocity_world = np.zeros(shape=(3,))
    diff_accel_norm = 0
    Q_norm = 0

    # Set up the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    line, = ax.plot([], [], [], 'r-')
    quiver = ax.quiver(0, 0, 0, 1, 0, 0)
    ax.set_xlim(-10e2, 10e2)
    ax.set_ylim(-10e2, 10e2)
    ax.set_zlim(-10e2, 10e2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.ion()

    for i in range(N):
        dt = timestamp[i]
        orientation.Dt = dt
        Q[i] = orientation.updateIMU(Q[i - 1], gyr=gyroscope[i], acc=accelerometer[i])
        R = Rotation.from_quat(Q[i]).as_matrix()
        accel_body = accelerometer[i]
        accel_world = (R @ accel_body.reshape(3, 1) - cal_err_lin_world.reshape(3, 1)).reshape(3, )
        velocity_world = velocity_world + accel_world * dt
        points[i] = points[i - 1] + velocity_world * dt

        diff_accel_norm = np.abs(np.linalg.norm(accel_body) - np.linalg.norm(accel_world))
        Q_norm += np.linalg.norm(Q[i])

        # Update the plot
        line.set_data(points[:i + 1, 0], points[:i + 1, 1])
        line.set_3d_properties(points[:i + 1, 2])

        # Update the heading arrow
        heading = R @ np.array([1, 0, 0])
        quiver.remove()
        quiver = ax.quiver(points[i, 0], points[i, 1], points[i, 2],
                           heading[0], heading[1], heading[2],
                           length=1, normalize=True, color='g')

        plt.title(f'Drone Position and Heading (Frame {i + 1}/{N})')
        plt.draw()
        plt.pause(0.01)

    print('||a_world - a_body||_avg=', diff_accel_norm / N, '\n', '||Q||_avg = ', Q_norm / N, '\n')
    plt.ioff()
    plt.show()
    return points, Q

def calc_position_from_IMU_2D(timestamp, gyroscope, accelerometer, cal_err_lin, cal_err_lin_world, cal_err_gyr):
    print("##-------Data parameters--------------------", '\n')
    gyroscope = gyroscope[:, :2] - cal_err_gyr[:2]  # Use only X and Y components
    accelerometer = accelerometer[:, :2] - cal_err_lin[:2]  # Use only X and Y components
    N = len(gyroscope)

    # Initialize 2D arrays
    points = np.zeros((N, 2))
    velocity = np.zeros(2)
    angle = 0  # Single angle for 2D rotation

    # Set up the 2D plot
    plt.figure(figsize=(10, 8))
    plt.ion()

    for i in range(1, N):
        dt = timestamp[i] - timestamp[i - 1]

        # Update angle using only z-axis of gyroscope (assuming it's the rotation around z-axis)
        angle += gyroscope[i, 1] * dt  # Assuming gyroscope[i, 1] is the rotation rate around z-axis

        # Create 2D rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        # Rotate accelerometer readings to world frame
        accel_world = rotation_matrix @ accelerometer[i] - cal_err_lin_world[:2]

        # Update velocity and position
        velocity += accel_world * dt
        points[i] = points[i - 1] + velocity * dt

        # Update the 2D plot
        plt.clf()  # Clear the current figure
        plt.plot(points[:i + 1, 0], points[:i + 1, 1], 'b-')  # Plot trajectory
        plt.plot(points[i, 0], points[i, 1], 'ro')  # Plot current position

        # Plot heading arrow
        heading = rotation_matrix @ np.array([1, 0])
        plt.arrow(points[i, 0], points[i, 1], heading[0], heading[1],
                  head_width=0.2, head_length=0.3, fc='g', ec='g')

        plt.xlim(min(points[:, 0]) - 1, max(points[:, 0]) + 1)
        plt.ylim(min(points[:, 1]) - 1, max(points[:, 1]) + 1)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'2D Drone Position and Heading (Frame {i + 1}/{N})')
        plt.grid(True)
        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()
    return points

def dead_reckoning_3d(timestamps, accel_data, gyro_data):
    dt = np.diff(timestamps)
    num_samples = len(timestamps)

    # Initialize arrays
    position = np.zeros((num_samples, 3))
    velocity = np.zeros((num_samples, 3))
    rotation_matrix = np.zeros((num_samples, 3, 3))
    rotation_matrix[0] = np.eye(3)  # Initial rotation matrix

    for i in range(1, num_samples):
        # Update rotation matrix using gyroscope data
        wx, wy, wz = gyro_data[i - 1] * dt[i - 1]
        dR = np.array([
            [1, -wz, wy],
            [wz, 1, -wx],
            [-wy, wx, 1]
        ])
        rotation_matrix[i] = rotation_matrix[i - 1] @ dR

        # Rotate acceleration to global frame
        accel_global = rotation_matrix[i] @ accel_data[i - 1]

        # Update velocity (trapezoidal integration)
        velocity[i] = velocity[i - 1] + 0.5 * (accel_global + rotation_matrix[i] @ accel_data[i]) * dt[i - 1]

        # Update position (trapezoidal integration)
        position[i] = position[i - 1] + 0.5 * (velocity[i - 1] + velocity[i]) * dt[i - 1]

    return position, velocity, rotation_matrix


def plot_trajectory_3d(position, rotation_matrix):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(position[:, 0], position[:, 1], position[:, 2], 'b-')
    ax.scatter(position[0, 0], position[0, 1], position[0, 2], c='g', s=100, label='Start')
    ax.scatter(position[-1, 0], position[-1, 1], position[-1, 2], c='r', s=100, label='End')

    # Plot orientation arrows at regular intervals
    num_arrows = 20
    indices = np.linspace(0, len(position) - 1, num_arrows, dtype=int)
    arrow_length = np.max(np.ptp(position, axis=0)) / 20

    for i in indices:
        p = position[i]
        R = rotation_matrix[i]

        # Plot arrows for each axis of the rotation matrix
        ax.quiver(p[0], p[1], p[2], R[0, 0], R[1, 0], R[2, 0], color='r', length=arrow_length)
        ax.quiver(p[0], p[1], p[2], R[0, 1], R[1, 1], R[2, 1], color='g', length=arrow_length)
        ax.quiver(p[0], p[1], p[2], R[0, 2], R[1, 2], R[2, 2], color='b', length=arrow_length)

    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title('3D Trajectory with Orientation')
    ax.legend()
    plt.show()

# Example usage:
# Assuming you have your data in numpy arrays:
# timestamps: 1D array of timestamps
# accel_data: Nx3 array of accelerometer data (x, y, z)
# gyro_data: Nx3 array of gyroscope data (x, y, z)

# position, velocity, rotation_matrix = dead_reckoning_3d(timestamps, accel_data, gyro_data)
# plot_trajectory_3d(position, rotation_matrix)