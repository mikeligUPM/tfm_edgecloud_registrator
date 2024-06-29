import os
import base64
import json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
import time
import numpy as np
import open3d as o3d

BROKER_IP = "192.168.56.101" # Local VM broker. For testing only.
BROKER_IP = "20.82.113.36" # Azure broker
BROKER_PORT = 1883 # Todo: Add security to MQTT
TOPIC = "cameraframes"

SEND_FREQUENCY = 1 # Time in seconds between sending messages

def create_k_dict_by_camera(filepath) -> dict:
    k_dict = {}
    K = np.eye(3)
    with open(filepath, "r") as f:
        data = json.load(f)

        for cam_number, camera in enumerate(data["cameras"]):
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
    # with open(file_path, 'rb') as f:
    #     png_bytes = f.read()
    # encoded_image = base64.b64encode(png_bytes).decode('utf-8')
    
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
        "target_model": "icp_p2l" # either "icp_p2p", "icp_p2l" or "geotrans"
    }
    
    client.publish(TOPIC, json.dumps(payload))
    print(f"Sent message to IoT Hub")
    time.sleep(1)

def main(client):
    # Directory where the PNG files are located
    directory = '../data/test_frame/'
    i=0
    k_dict = create_k_dict_by_camera("cam_params.json")
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        # print(f"\n## PROCESSING FILE: {filename} ##\n")
        if filename.endswith(".png"):
            # Extract device ID and type of image (color or depth) from filename
            parts = filename.split('_')
            device_id = parts[0]
            image_type = parts[1]  # 'color' or 'depth'
            frame_id = parts[2].split(".")[0]
            
            # Encode the PNG file to base64
            encoded_image = encode_png_to_base64(os.path.join(directory, filename))
            print(f"{i}: Len encoded_image = {len(encoded_image)}\n\n")
            # Assuming all images are from frame 'f0001'
            camera_name = device_id
            
            # Send the message to IoT Hub
            if image_type == 'color':
                encoded_color_image = encoded_image
            elif image_type == 'depth':
                encoded_depth_image = encoded_image
            
            # Check if both color and depth images are ready to be sent
            if 'encoded_color_image' in locals() and 'encoded_depth_image' in locals():
                build_publish_encoded_msg(client, frame_id, camera_name, 
                                          encoded_color_image, encoded_depth_image, k_dict[camera_name])
                # Reset variables for the next pair of images
                del encoded_color_image, encoded_depth_image
                
                time.sleep(SEND_FREQUENCY)
            i+=1
            
    print("END")

if __name__ == "__main__":
    try:
        # Conexión al broker MQTT
        client = mqtt.Client()
        client.connect(BROKER_IP, BROKER_PORT)
    except Exception as e:
        print(f"Could not connect to broker: {e}")
    else:
        # Inicio de la publicación de datos
        print("Connected. Starting publish")
        main(client)