import numpy as np
import pyrealsense2 as rs

def get_3d_point(depth_frame, x, y):
    depth = depth_frame.get_distance(x, y)
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return point  # [x, y, z] in meters

def match_detections(detections_1, detections_2, depth_frame_1, depth_frame_2):
    matched_detections = []
    for det_1 in detections_1:
        class_id_1, bbox_1, confidence_1 = det_1
        x1, y1, w1, h1 = bbox_1
        center_x1, center_y1 = x1 + w1 // 2, y1 + h1 // 2
        point_1 = get_3d_point(depth_frame_1, center_x1, center_y1)

        for det_2 in detections_2:
            class_id_2, bbox_2, confidence_2 = det_2
            if class_id_1 != class_id_2:
                continue
            x2, y2, w2, h2 = bbox_2
            center_x2, center_y2 = x2 + w2 // 2, y2 + h2 // 2
            point_2 = get_3d_point(depth_frame_2, center_x2, center_y2)

            distance = np.linalg.norm(np.array(point_1) - np.array(point_2))
            if distance < 0.1:  # Adjust the threshold as needed
                matched_detections.append((bbox_1, bbox_2))
                break
    return matched_detections
