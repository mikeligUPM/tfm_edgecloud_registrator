import paho.mqtt.client as mqtt
import base64
import numpy as np
import open3d as o3d
import cv2
import json
import threading
from threading import Timer
import time

from constants import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, CAMERA_COUNT, VOXEL_SIZE
from registrator_icp import icp_p2p_registration, icp_p2l_registration
from registrator_geotrans import geotrans_registration
from blob_handler import save_and_upload_pcd

batch_timeout = 10

registration_methods = {
    "icp_p2p": icp_p2p_registration,
    "icp_p2l": icp_p2l_registration,
    "geotransformer": geotrans_registration,
}

received_frames_dict = {}
received_frames_lock = threading.Lock()


def calculate_voxel_size(point_cloud, percentage):
    """
    Calculate recommended voxel size based on a percentage of the bounding box size of an Open3D point cloud.

    Parameters:
    - point_cloud (open3d.geometry.PointCloud): Open3D point cloud object.
    - percentage (float): Percentage of the bounding box size that should be used as the voxel size.

    Returns:
    - float: Recommended voxel size.
    """
    # Calculate bounding box dimensions
    bbox = point_cloud.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_max_bound() - bbox.get_min_bound()

    # Calculate voxel size based on percentage of bounding box size
    voxel_size = max(bbox_size) * percentage / 100.0

    return voxel_size

def create_pc_from_encoded_data(color_encoded, depth_encoded, K):
    color_image_data = base64.b64decode(color_encoded)
    depth_image_data = base64.b64decode(depth_encoded)
    print(f"len depth_image_data: {len(depth_image_data)}\n")

    # Add a single byte if the length of depth_image_data is odd
    if len(depth_image_data) % 2 != 0:
        depth_image_data += b'\x00'
        print(f"Adjusted len depth_image_data: {len(depth_image_data)}\n")

    if len(color_image_data) % 2 != 0:
        color_image_data += b'\x00'
        print(f"Adjusted len depth_image_data: {len(depth_image_data)}\n")

    try:
        color_image = cv2.imdecode(np.frombuffer(color_image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        depth_image = cv2.imdecode(np.frombuffer(depth_image_data, dtype=np.uint16), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error loading image: {e}   || len depth_image_data: {len(depth_image_data)}  len color: {len(color_image_data)}\n")
        return None

    depth_raw = o3d.geometry.Image(depth_image.astype(np.float32) / 1000.0)
    color_raw = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(color_image.shape[1], color_image.shape[0], K[0][0], K[1][1], K[0][2], K[1][2])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    print(f"PCD len before downsampling {len(pcd.points)}")
    recommended_box_size = calculate_voxel_size(pcd, 5)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"PCD len after downsampling {len(pcd.points)}")
    pcd.estimate_normals()
    return pcd

def process_frame(message_json_list, frame_id):
    pcd_list = []
    i = 1
    # print(f"Entered process_frame with frame_id = {frame_id}")
    # print("Going to sleep 15 seconds")
    # time.sleep(15)
    # print("Woke up")
    # return
    for message in message_json_list:
        print(f"Processing message {i}")
        camera_name = message.get('camera_name')
        enc_c = message.get('enc_c')
        enc_d = message.get('enc_d')
        ts = message.get('send_ts')
        K = message.get('K')


        pc = create_pc_from_encoded_data(enc_c, enc_d, K)
        if pc is None:
            print(f"Error creating point cloud for camera {camera_name} with frame {frame_id}")
            i += 1
            continue
        pcd_list.append(pc)
        i += 1
        blob_name_reg = f"P2P_test__{frame_id}_{i}.ply"  #testing only
        save_and_upload_pcd(pc, blob_name_reg) #testing only
    x = input("TESTING CONTROL C NOW")
    print(f"PCD List Length = {len(pcd_list)}")


    target_registration = message_json_list[0].get('target_model')
    registration_func = registration_methods.get(target_registration)
    if registration_func:
        final_fused_point_cloud = registration_func(pcd_list)
    else:
        raise ValueError(f"Unknown registration method: {target_registration}")
    blob_name_reg = f"reg__{frame_id}__{target_registration}.ply"
    save_and_upload_pcd(final_fused_point_cloud, blob_name_reg)

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_batch_timeout(frame_id):
    with received_frames_lock:
        print(f"4_- Timeout for {frame_id} detected")
        if frame_id in received_frames_dict and received_frames_dict[frame_id][0]:
            received_frames_dict[frame_id][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(frame_id)[0]
            threading.Thread(target=process_frame, args=(frame_data_copy, frame_id)).start()

def process_message(msg):
    message = json.loads(msg.payload)
    frame_id = message.get('frame_id')
    print(f"Received MSG with FrameID = {frame_id}")

    with received_frames_lock:
        if frame_id not in received_frames_dict:
            received_frames_dict[frame_id] = ([], Timer(batch_timeout, on_batch_timeout, args=[frame_id]))
            received_frames_dict[frame_id][1].start()  # Start the timer

        received_frames_dict[frame_id][0].append(message)
        if len(received_frames_dict[frame_id][0]) == CAMERA_COUNT:
            print(f"received_frames_dict[frame_id] == CAMERA_COUNT GOING TO PROCESS {frame_id}")
            received_frames_dict[frame_id][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(frame_id)[0]
            threading.Thread(target=process_frame, args=(frame_data_copy, frame_id)).start()
        else:
            # As a new event has arrived, reset timer
            if received_frames_dict[frame_id][1] is not None:
                received_frames_dict[frame_id][1].cancel()
            received_frames_dict[frame_id] = (
                received_frames_dict[frame_id][0],
                Timer(batch_timeout, on_batch_timeout, args=[frame_id])
            )
            received_frames_dict[frame_id][1].start()

def on_message(client, userdata, msg):
    threading.Thread(target=process_message, args=(msg,)).start()

def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

main()