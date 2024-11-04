import cv2
import numpy as np
from camera_stream import start_cameras
#yfrom object_detection import detect_objects
from depth_matching import match_detections
from evaluation import evaluate
from calibrate_cameras import main as calibrate_main


def show_camera_frames(pipeline_1, pipeline_2, pipeline_3):
    # Display frames from each camera to identify them
    for _ in range(30):  # Show 30 frames
        frames_1 = pipeline_1.wait_for_frames()
        frames_2 = pipeline_2.wait_for_frames()
        frames_3 = pipeline_3.wait_for_frames()

        color_frame_1 = frames_1.get_color_frame()
        color_frame_2 = frames_2.get_color_frame()
        color_frame_3 = frames_3.get_color_frame()

        if not color_frame_1 or not color_frame_2 or not color_frame_3:
            continue

        color_image_1 = np.asanyarray(color_frame_1.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        color_image_3 = np.asanyarray(color_frame_3.get_data())

        cv2.imshow('Camera 1', color_image_1)
        cv2.imshow('Camera 2', color_image_2)
        cv2.imshow('Camera 3', color_image_3)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    # Ask if the user wants to recalibrate the cameras
    recalibrate = input("Do you want to recalibrate the cameras? (yes/no): ").strip().lower()
    if recalibrate == 'yes':
        calibrate_main()

    # Start the cameras
    pipeline_1, pipeline_2, pipeline_3 = start_cameras()

    # Show frames from the cameras to identify them
    show_camera_frames(pipeline_1, pipeline_2, pipeline_3)

    try:
        while True:
            frames_1 = pipeline_1.wait_for_frames()
            frames_2 = pipeline_2.wait_for_frames()
            frames_3 = pipeline_3.wait_for_frames()

            depth_frame_1 = frames_1.get_depth_frame()
            color_frame_1 = frames_1.get_color_frame()
            depth_frame_2 = frames_2.get_depth_frame()
            color_frame_2 = frames_2.get_color_frame()
            depth_frame_3 = frames_3.get_depth_frame()
            color_frame_3 = frames_3.get_color_frame()

            if not depth_frame_1 or not color_frame_1 or not depth_frame_2 or not color_frame_2 or not depth_frame_3 or not color_frame_3:
                continue

            color_image_1 = np.asanyarray(color_frame_1.get_data())
            color_image_2 = np.asanyarray(color_frame_2.get_data())
            color_image_3 = np.asanyarray(color_frame_3.get_data())

            detections_1 = detect_objects(color_image_1)
            detections_2 = detect_objects(color_image_2)
            detections_3 = detect_objects(color_image_3)

            matched_detections = match_detections(detections_1, detections_2, depth_frame_1, depth_frame_2)

            for bbox_1, bbox_2 in matched_detections:
                x1, y1, w1, h1 = bbox_1
                x2, y2, w2, h2 = bbox_2
                color = (0, 255, 0)
                cv2.rectangle(color_image_1, (x1, y1), (x1 + w1, y1 + h1), color, 2)
                cv2.rectangle(color_image_2, (x2, y2), (x2 + w2, y2 + h2), color, 2)

            combined_images = np.vstack((color_image_1, color_image_2))
            cv2.imshow('Combined Detections', combined_images)

            single_camera_images = color_image_3
            for (class_id, bbox, confidence) in detections_3:
                x, y, w, h = bbox
                color = (0, 255, 0)
                cv2.rectangle(single_camera_images, (x, y), (x + w, y + h), color, 2)

            cv2.imshow('Single Camera Detections', single_camera_images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline_1.stop()
        pipeline_2.stop()
        pipeline_3.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
