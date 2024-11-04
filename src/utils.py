import cv2
import numpy as np

def calibrate_camera(image_files):
    obj_points = []
    img_points = []

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for image_file in image_files:
        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return camera_matrix, dist_coeffs

def stereo_calibrate(image_files_1, image_files_2, camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2):
    obj_points = []
    img_points_1 = []
    img_points_2 = []

    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    for file_1, file_2 in zip(image_files_1, image_files_2):
        img_1 = cv2.imread(file_1)
        img_2 = cv2.imread(file_2)
        gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
        gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
        ret_1, corners_1 = cv2.findChessboardCorners(gray_1, (9, 6), None)
        ret_2, corners_2 = cv2.findChessboardCorners(gray_2, (9, 6), None)
        if ret_1 and ret_2:
            obj_points.append(objp)
            img_points_1.append(corners_1)
            img_points_2.append(corners_2)

    _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(obj_points, img_points_1, img_points_2,
        camera_matrix_1, dist_coeffs_1, camera_matrix_2, dist_coeffs_2, gray_1.shape[::-1], None, None, None, None,
        cv2.CALIB_FIX_INTRINSIC)

    return R, T
