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

BROKER_IP = "20.82.113.36"  # Azure broker
BROKER_PORT = 1883  # Todo: Add security to MQTT
TOPIC = "cameraframes"

SEND_FREQUENCY = 1  # Time in seconds between sending messages

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


# Function to construct and send message to IoT Hub
def build_publish_encoded_msg(client, frame_id, camera_name, encoded_color_image, encoded_depth_image, K):
    dt_now = datetime.now(tz=timezone.utc)
    send_ts = round(dt_now.timestamp() * 1000)

    # encoded_color_image = "test"
    # encoded_depth_image = "test"
    
    payload = {
        "frame_id": frame_id,
        "camera_name": camera_name,
        "enc_c": encoded_color_image,
        "enc_d": encoded_depth_image,
        "K": K,
        "send_ts": send_ts,
        "target_model": "icp_p2l"  # either "icp_p2p", "icp_p2l" or "geotransformer"
    }

    client.publish(TOPIC, json.dumps(payload))
    print(f"Sent message to IoT Hub: {payload}")


# Function to process a frame
def process_frame(client, k_dict, camera_dir, frame_id):
    camera_name = camera_dir.split("\\")[-1]
    color_filename = f"{camera_name}_color_{frame_id}.png"
    depth_filename = f"{camera_name}_depth_{frame_id}.png"
    color_path = os.path.join(camera_dir, color_filename)
    depth_path = os.path.join(camera_dir, depth_filename)

    encoded_color_image = encode_png_to_base64(color_path)
    encoded_depth_image = encode_png_to_base64(depth_path)

    camera_name = os.path.basename(camera_dir)

    build_publish_encoded_msg(client, frame_id, camera_name,
                              encoded_color_image, encoded_depth_image, k_dict[camera_name])


# Main function to control the flow
def main(client):
    # base_directory = 'data\\png_data'
    base_directory = 'data\\png_data'
    k_dict = create_k_dict_by_camera("cam_params.json")

    camera_dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]
    while True:
        chosen_frame = random.choice(range(1, 219))  # Assuming frames are from 1 to 219
        print(f"Chosen frame: f{chosen_frame:04d}")

        threads = []
        for chosen_camera_dir in camera_dirs:
            print(f"Chosen camera directory: {chosen_camera_dir}")
            thread = threading.Thread(target=process_frame, args=(client, k_dict, chosen_camera_dir, f"f{chosen_frame:04d}"))
            threads.append(thread)
            thread.start()

        time.sleep(5)  # Wait for 5 seconds before choosing another frame and closing threads
        # for thread in threads:
        #     thread.join()  # Wait for all threads to complete


if __name__ == "__main__":
    try:
        # Connection to MQTT broker
        client = mqtt.Client()
        client.connect(BROKER_IP, BROKER_PORT)
    except Exception as e:
        print(f"Could not connect to broker: {e}")
    else:
        # Starting data publication
        print("Connected. Starting publish")
        main(client)
