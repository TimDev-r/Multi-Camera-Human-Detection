import cv2
import pyrealsense2 as rs
import numpy as np
import threading
from time import sleep

def capture_images():
    # Initialize the context
    ctx = rs.context()
    devices = ctx.query_devices()

    # Ensure we have at least three devices
    num_attempts = 10
    attempt = 0
    while len(devices) < 3:
        print(attempt)
        sleep(0.2)
        devices = ctx.query_devices()
        if (attempt := attempt + 1) > num_attempts:
            raise RuntimeError("At least three cameras are required.")

    # List device serial numbers
    device_serials = []
    for device in devices:
        device_serials.append(device.get_info(rs.camera_info.serial_number))
    print("Detected device serial numbers:", device_serials)

    # Initialize pipelines for three cameras
    pipeline_1 = rs.pipeline()
    pipeline_2 = rs.pipeline()
    pipeline_3 = rs.pipeline()

    config_1 = rs.config()
    config_2 = rs.config()
    config_3 = rs.config()

    # Enable streams and devices by serial number
    config_1.enable_device(device_serials[0])
    config_2.enable_device(device_serials[1])
    config_3.enable_device(device_serials[2])

    config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_3.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the pipelines
    pipeline_1.start(config_1)
    pipeline_2.start(config_2)
    pipeline_3.start(config_3)

    def capture_from_camera(pipeline, index):
        for i in range(20):  # Capture 20 images
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imwrite(f"calibration_images/camera{index}_image_{i}.png", color_image)
            cv2.imshow(f'Camera {index}', color_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Create threads for each camera
    thread_1 = threading.Thread(target=capture_from_camera, args=(pipeline_1, 1))
    thread_2 = threading.Thread(target=capture_from_camera, args=(pipeline_2, 2))
    thread_3 = threading.Thread(target=capture_from_camera, args=(pipeline_3, 3))

    # Start the threads
    thread_1.start()
    thread_2.start()
    thread_3.start()

    # Wait for all threads to finish
    thread_1.join()
    thread_2.join()
    thread_3.join()

    # Cleanup
    pipeline_1.stop()
    pipeline_2.stop()
    pipeline_3.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_images()
