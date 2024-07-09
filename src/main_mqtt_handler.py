import paho.mqtt.client as mqtt
import base64
import numpy as np
import open3d as o3d
import cv2
import json
import threading
from threading import Timer
import time




from logger_config import logger
from constants import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, CAMERA_COUNT
from helper_funs import get_config
# from registrator_icp import icp_p2p_registration, icp_p2l_registration
from registrator_icp_ransac import icp_p2p_registration_ransac, icp_p2l_registration_ransac
from registrator_geotrans import geotrans_registration
from blob_handler import save_and_upload_pcd

registration_methods = {
    # "unusued_icp_p2p": icp_p2p_registration,
    # "unusued_icp_p2l": icp_p2l_registration,
    0: icp_p2p_registration_ransac,
    1: icp_p2l_registration_ransac,
    2: geotrans_registration,
}

registration_names_from_id = {
    0: "icp_p2p_ransac",
    1: "icp_p2l_ransac",
    2: "geotransformer"
}

batch_timeout = 60 # seconds

received_frames_dict = {}
received_frames_lock = threading.Lock()


def create_pc_from_encoded_data(color_encoded, depth_encoded, K, target_ds):
    color_image_data = base64.b64decode(color_encoded)
    depth_image_data = base64.b64decode(depth_encoded)
    logger.debug(f"len depth_image_data: {len(depth_image_data)}\n")

    # Add a single byte if the length of depth_image_data is odd
    if len(depth_image_data) % 2 != 0:
        depth_image_data += b'\x00'
        logger.debug(f"Adjusted len depth_image_data: {len(depth_image_data)}\n")

    if len(color_image_data) % 2 != 0:
        color_image_data += b'\x00'
        logger.debug(f"Adjusted len depth_image_data: {len(depth_image_data)}\n")

    try:
        color_image = cv2.imdecode(np.frombuffer(color_image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
        depth_image = cv2.imdecode(np.frombuffer(depth_image_data, dtype=np.uint16), cv2.IMREAD_UNCHANGED)
    except Exception as e:
        logger.error(f"Error loading image: {e}   || len depth_image_data: {len(depth_image_data)}  len color: {len(color_image_data)}\n")
        return None

    depth_raw = o3d.geometry.Image(depth_image.astype(np.float32) / 1000.0)
    color_raw = o3d.geometry.Image(color_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(color_image.shape[1], color_image.shape[0], K[0][0], K[1][1], K[0][2], K[1][2])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    logger.debug(f"PCD len before downsampling {len(pcd.points)}")
    vox_size, _ = get_config(target_ds)
    logger.debug(f"[TEST] Voxel size: {vox_size}")
    pcd = pcd.voxel_down_sample(voxel_size=vox_size)
    logger.debug(f"PCD len after downsampling {len(pcd.points)}")
    pcd.estimate_normals()
    return pcd

def process_frame(message_json_list, frame_id):
    pcd_list = []
    i = 1
    for message in message_json_list:
        logger.info(f"[TS] Frame [{frame_id}] Processing message {i} / {len(message_json_list)}")
        camera_name = message.get('camera_name')
        enc_c = message.get('enc_c')
        enc_d = message.get('enc_d')
        ts = message.get('send_ts')
        K = message.get('K')
        target_ds = message.get('ds')


        pc = create_pc_from_encoded_data(enc_c, enc_d, K, target_ds)
        if pc is None:
            logger.error(f"Error creating point cloud for camera {camera_name} with frame {frame_id}")
            i += 1
            continue
        logger.info(f"[TS] Frame [{frame_id}] PCD created for camera {camera_name}")
        pcd_list.append(pc)
        i += 1
        # blob_name_reg = f"P2P_test__{frame_id}_{i}.ply"  #testing only
        # save_and_upload_pcd(pc, blob_name_reg) #testing only
    logger.info(f"Frame [{frame_id}] PCD List Length = {len(pcd_list)}")


    target_registration = message_json_list[0].get('reg')
    registration_func = registration_methods.get(target_registration)
    if registration_func:
        final_fused_point_cloud = registration_func(pcd_list)
    else:
        logger.warning(f"Frame [{frame_id}] Unknown registration method: {target_registration}")
    if final_fused_point_cloud:
        logger.debug(f"Frame [{frame_id}] Registration successful")
        
        try:
            reg_name = registration_names_from_id.get(target_registration)
        except Exception as e:
            reg_name = "unknown"
            logger.error(f"Error getting registration name: {e}")
        blob_name_reg = f"reg__{frame_id}__{reg_name}.ply"
        save_and_upload_pcd(final_fused_point_cloud, blob_name_reg)
    else:
        logger.info(f"Frame [{frame_id}] Final PCD is None. Please check error logs.")


def on_batch_timeout(frame_id):
    with received_frames_lock:
        logger.info(f"Timeout for frame {frame_id} detected")
        if frame_id in received_frames_dict and received_frames_dict[frame_id][0]:
            received_frames_dict[frame_id][1].cancel()  # Stop the timer
            frame_data_copy = received_frames_dict.pop(frame_id)[0]
            threading.Thread(target=process_frame, args=(frame_data_copy, frame_id)).start()

def process_message(msg):
    message = json.loads(msg.payload)
    frame_id = message.get('frame_id')
    logger.info(f"[TS] Received MSG with FrameID = {frame_id}")

    with received_frames_lock:
        if frame_id not in received_frames_dict:
            received_frames_dict[frame_id] = ([], Timer(batch_timeout, on_batch_timeout, args=[frame_id]))
            received_frames_dict[frame_id][1].start()  # Start the timer

        received_frames_dict[frame_id][0].append(message)
        if len(received_frames_dict[frame_id][0]) == CAMERA_COUNT:
            logger.info(f"Batch full for frame {frame_id}")
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


### Mqtt funs
def on_connect(client, userdata, flags, rc):
    logger.info(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    threading.Thread(target=process_message, args=(msg,)).start()

### Main
def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

main()