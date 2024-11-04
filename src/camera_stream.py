import pyrealsense2 as rs
import numpy as np

def start_cameras():
    pipeline_1 = rs.pipeline()
    pipeline_2 = rs.pipeline()
    pipeline_3 = rs.pipeline()

    config_1 = rs.config()
    config_2 = rs.config()
    config_3 = rs.config()

    devices = rs.context().devices
    device_1 = devices[0]
    device_2 = devices[1]
    device_3 = devices[2]

    config_1.enable_device(device_1.get_info(rs.camera_info.serial_number))
    config_2.enable_device(device_2.get_info(rs.camera_info.serial_number))
    config_3.enable_device(device_3.get_info(rs.camera_info.serial_number))

    config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    config_3.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config_3.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    pipeline_1.start(config_1)
    pipeline_2.start(config_2)
    pipeline_3.start(config_3)

    return pipeline_1, pipeline_2, pipeline_3
