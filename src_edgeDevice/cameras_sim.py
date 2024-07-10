import os
import base64
import json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
import time
import numpy as np
import open3d as o3d
import random
import threading

from logger_config import logger

# BROKER_IP = "20.82.113.36"  # Azure broker
BROKER_IP = "172.205.149.105"  # Azure broker
BROKER_PORT = 1883  # Todo: Add security to MQTT
TOPIC = "cameraframes"
SEND_FREQUENCY = 1  # Time in seconds between sending messages

logger.info(f"Camera simulation started with:\nBROKER_IP: {BROKER_IP}\nBROKER_PORT: {BROKER_PORT}\nTOPIC: {TOPIC}\nSEND_FREQUENCY: {SEND_FREQUENCY}")

K = [
    [585.0, 0.0, 320.0],
    [0.0, 585.0, 240.0],
    [0.0, 0.0, 1.0]
]

dataset_name_from_id = {
    0: "3DMatch",
    1: "ownData"
}

dataset_id_from_name = {
    "3DMatch": 0,
    "ownData": 1
}

def save_json_to_local(payload, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(payload, json_file, indent=4)

def create_k_dict_by_camera(filepath) -> dict:
    k_dict = {}
    K = np.eye(3)
    with open(filepath, "r") as f:
        data = json.load(f)

        for _, camera in enumerate(data["cameras"]):
            # Extract camera parameters
            resolution = camera["Resolution"]
            focal = camera["Focal"]
            principal_point = camera["Principle_point"]
            camera_name = camera["Name"]

            # Create PinholeCameraIntrinsic object
            K = o3d.camera.PinholeCameraIntrinsic(
                width=resolution[0],
                height=resolution[1],
                fx=focal[0],
                fy=focal[1],
                cx=principal_point[0],
                cy=principal_point[1]
            )
            k_dict[camera_name] = K.intrinsic_matrix.tolist()
    return k_dict


# Function to encode PNG file to base64
def encode_png_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_message_size(payload):
    # Calculate size of JSON-encoded payload in bytes
    return len(json.dumps(payload))

# Function to construct and send message to IoT Hub
def build_publish_encoded_msg(client, frame_id, camera_name, encoded_color_image, encoded_depth_image, K, dataset_id):
    dt_now = datetime.now(tz=timezone.utc)
    send_ts = round(dt_now.timestamp() * 1000)

    # encoded_color_image = "test_color"
    # encoded_depth_image = "test_depth"
    
    ## Payload options
    ## "reg": target registration method
    ### 0: icp_p2p_ransac
    ### 1: icp_p2l_ransac
    ### 2: geotransformer
    
    ## "ds": dataset name
    ### 0: 3DMatch
    ### 1: Own data
    
    payload = {
        "frame_id": frame_id,
        "camera_name": camera_name,
        "enc_c": encoded_color_image,
        "enc_d": encoded_depth_image,
        "K": K,
        "reg": 0,
        "ds": dataset_id,
        "send_ts": send_ts # UTC timestamp
    }
    # Calculate message size
    message_size = get_message_size(payload)
    logger.info(f"Message size: {message_size} bytes")
    
    
    # save_json_to_local(payload, f"encoded_jsons\\{camera_name}_{frame_id}.json")
    client.publish(TOPIC, json.dumps(payload), qos=1)
    # logger.debug(f"Test payload: {payload}")
    logger.info(f"[TS] Camera [{camera_name}] Sent message to IoT Hub: {frame_id}")
    # time.sleep(2)


def get_image_path(camera_dir, frame_id):
    if '3DMatch' in camera_dir:
        color_filename = f"{frame_id}_color.png"
        depth_filename = f"{frame_id}_depth.png"
    else:
        camera_name = camera_dir.split("\\")[-1]
        color_filename = f"{camera_name}_color_{frame_id}.png"
        depth_filename = f"{camera_name}_depth_{frame_id}.png"
    return os.path.join(camera_dir, color_filename), os.path.join(camera_dir, depth_filename)
    

# Function to process a frame
def process_frame(client, k_dict, camera_dir, frame_id, dataset_id):
    camera_name = os.path.basename(camera_dir)
    logger.info(f"[{frame_id}] Camera {camera_name} INIT")
    color_path, depth_path = get_image_path(camera_dir, frame_id)

    encoded_color_image = encode_png_to_base64(color_path)
    encoded_depth_image = encode_png_to_base64(depth_path)


    if isinstance(k_dict, dict):
        k_list = k_dict[camera_name]
    elif isinstance(k_dict, list):
        k_list = k_dict
    build_publish_encoded_msg(client, frame_id, camera_name,
                              encoded_color_image, encoded_depth_image, k_list, dataset_id)
    logger.info(f"[{frame_id}] Camera {camera_name} END")
        
    
       
       
# Main function to control the flow
def start_cam_simulation(client, base_directory, dataset_id, send_freq=3):
    exit_sim = False
    if dataset_id == 1: # Own dataset
        k_dict = create_k_dict_by_camera("cam_params.json")
    else:
        k_dict = K # 3DMatch dataset
    try:
        while not exit_sim:
            i = 0
            camera_dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

            chosen_frame = random.choice(range(1, 219))  # Assuming frames are from 1 to 219
            chosen_frame = 125
            
            if dataset_id == 0: # 3DMatch
                chosen_frame_str = f"frame-{chosen_frame:06d}"
            elif dataset_id == 1: # own data
                chosen_frame_str = f"f{chosen_frame:04d}"
            else:
                logger.error(f"Invalid dataset_id: {dataset_id}")
                exit_sim = True
                break
            
            logger.info(f"Using {dataset_name_from_id[dataset_id]} dataset")
            logger.info(f"Chosen frame: f{chosen_frame:04d}")

            threads = []
            for chosen_camera_dir in camera_dirs:
                # logger.debug((f"Chosen camera directory: {chosen_camera_dir}")
                thread = threading.Thread(target=process_frame, args=(client, k_dict, chosen_camera_dir, chosen_frame_str, dataset_id))
                threads.append(thread)
                thread.start()
                time.sleep(0.1)

            time.sleep(send_freq)  # Wait for N seconds before choosing another frame
            x = input("continue? ")
    except KeyboardInterrupt:
        exit_sim = True

def ds_selection_prompt():
    logger.info("Select the dataset you want to simulate:\n'0': 3DMatch\n'1': Own dataset")
    ds = input("Selection: ")
    if ds == '0':
        return '..\\data\\3DMatch', 0
    else:
        return '..\\data\\png_data', 1
    
def on_publish(client, userdata, mid):
    logger.info(f"Message published successfully with MID: {mid}")

if __name__ == "__main__":
    base_directory, dataset_id = ds_selection_prompt()
    logger.info(f"Selected dataset dir: {base_directory}  dataset: {dataset_id}")
    try:
        # Connection to MQTT broker
        client = mqtt.Client()
        client.on_publish = on_publish
        client.connect(BROKER_IP, BROKER_PORT)
        client.loop_start()
    except Exception as e:
        logger.error(f"Could not connect to broker: {e}")
    else:
        # Starting data publication
        logger.info("Connected. Starting publish")
        start_cam_simulation(client, base_directory, dataset_id ,send_freq=SEND_FREQUENCY)
        logger.info("Simulation ended")
