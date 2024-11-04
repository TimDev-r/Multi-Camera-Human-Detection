import cv2
import numpy as np
from utils import calibrate_camera, stereo_calibrate


def main():
    image_files_1 = [f"calibration_images/camera1_image_{i}.png" for i in range(20)]
    image_files_2 = [f"calibration_images/camera2_image_{i}.png" for i in range(20)]

    camera_matrix_1, dist_coeffs_1 = calibrate_camera(image_files_1)
    camera_matrix_2, dist_coeffs_2 = calibrate_camera(image_files_2)

    R, T = stereo_calibrate(image_files_1, image_files_2, camera_matrix_1, dist_coeffs_1, camera_matrix_2,
                            dist_coeffs_2)

    np.savez("calibration_data.npz", camera_matrix_1=camera_matrix_1, dist_coeffs_1=dist_coeffs_1,
             camera_matrix_2=camera_matrix_2, dist_coeffs_2=dist_coeffs_2, R=R, T=T)


if __name__ == "__main__":
    main()
