import sys
from time import sleep
from threading import Lock, Thread
import numpy as np
import cv2
import torch

import pyzed
import pyzed.sl as sl

import yolov7
from yolov7.hubconf import create

from filter import Filter
from InverseKinematics import IK
import servo


BIRD = 14
# BIRD = 0

lock = Lock()
run_signal = False
exit_signal = False


def results_to_detections(results):
    detections = []
    for det in results.xyxy[0]:
        if det[5] == BIRD:
            detection = sl.CustomBoxObjectData()
            detection.unique_object_id = sl.generate_unique_id()
            detection.probability = det[4]
            detection.label = det[5]
            box = np.zeros((4, 2))
            xmin = det[0]
            xmax = det[2]
            ymin = det[1]
            ymax = det[3]

            box[0][0] = xmin
            box[0][1] = ymin

            box[1][0] = xmax
            box[1][1] = ymin

            box[2][0] = xmin
            box[2][1] = ymax

            box[3][0] = xmax
            box[3][1] = ymax

            detection.bounding_box_2d = box
            detection.is_grounded = False
            detections.append(detection)

    return detections

def torch_thread():
    global lock, exit_signal, run_signal, cv_img, results

    model = yolov7.load('yolov7.pt', device="cuda:0", trace=False, size=1280)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            results = model(cv_img)

            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global lock, exit_signal, run_signal, cv_img, results

    model_thread = Thread(target=torch_thread)
    model_thread.start()

    # Create a ZED camera object
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 15
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 20
    init_params.sdk_verbose = True

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit()

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    detection_parameters = sl.ObjectDetectionParameters()
    detection_parameters.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    detection_parameters.image_sync = True
    detection_parameters.enable_tracking = True
    detection_parameters.enable_mask_output = True

    print("Object Detection: Loading Module...")
    err = zed.enable_object_detection(detection_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Error {}, exit program".format(err))
        zed.close()
        exit()

    runtime_params = sl.RuntimeParameters()

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 45

    image_left = sl.Mat()
    cam_w_pose = sl.Pose()

    filter = Filter()
    ik = IK()

    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS and not exit_signal:
        lock.acquire()
        zed.retrieve_image(image_left, sl.VIEW.LEFT)
        cv_img = image_left.get_data()
        lock.release()
        run_signal = True

        # -- Detection running on the other thread
        while run_signal:
            sleep(0.001)

        # Wait for detections
        lock.acquire()
        # -- Ingest detections
        detections = results_to_detections(results)
        zed.ingest_custom_box_objects(detections)
        lock.release()

        objects = sl.Objects()
        err = zed.retrieve_objects(objects, obj_runtime_param)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Error {}, exit program".format(err))
            zed.close()
            exit()

        target = filter.getTarget(objects)
        if target is not None:
            # if all target.position are not nan
            if not np.isnan(target.position).any():
                servo_target = ik.calc(*target.position)
                servo.move_servos(servo_target["rotate"], servo_target["tilt"])

        # -- Draw detections
        for obj in objects.object_list:
            # continue if any of the position is nan
            if np.isnan(obj.position).any():
                continue
            color = (0, 255, 0)
            if obj.id == target.id:
                color = (0, 0, 255)
            cv2.rectangle(cv_img, (int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1])), (int(obj.bounding_box_2d[2][0]), int(obj.bounding_box_2d[2][1])), color, 2)
            # display text
            text = f"{obj.confidence}%"
            cv2.putText(cv_img, text, (int(obj.bounding_box_2d[0][0]), int(obj.bounding_box_2d[0][1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Robo-Scarecrow", cv_img)
        key = cv2.waitKey(10)
        if key == 27:
            exit_signal = True
            break

    cv2.destroyAllWindows()
    zed.disable_object_detection()
    zed.close()
    print("Exiting...")
    exit()


if __name__ == "__main__":
    with torch.no_grad():
        main()

# -0.33 0.22 -0.50