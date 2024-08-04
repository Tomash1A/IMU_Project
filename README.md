# IMU_Project
 Implemnting dead-reckoning algorythm (no gps localization), using Mahony's ahrs filter, on an IMU dataset from UZH (https://fpv.ifi.uzh.ch/datasets/).
 Instructions:
 1. Download all files (including data_imu.txt, and groundtruth.txt)/ clone repositpry into a project in your computer.
 2. Rename the path of imu_data.txt and groundtruth.txt  (trajectory_calc.py, line 15-16), to the path of the files in your computer.
 3. ensure you have he following python libraries: ahrs, scipy, numpy, matplotlib, docutils, mpl_toolkits, zipfile
 4. Run trajectory_calc.py file.
