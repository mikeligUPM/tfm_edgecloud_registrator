import paho.mqtt.client as mqtt
import base64
import numpy as np
import open3d as o3d
import cv2
import json
from multiprocessing import Process, Manager
import copy

from constants import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, CAMERA_COUNT, VOXEL_SIZE
from registrator_icp import icp_p2p_registration, icp_p2l_registration
from registrator_geotrans import geotrans_registration
from blob_handler import save_and_upload_pcd

registration_methods = {
    "icp_p2p": icp_p2p_registration,
    "icp_p2l": icp_p2l_registration,
    "geotransformer": geotrans_registration,
}

received_frames_dict = {}



def create_pc_from_encoded_data(color_encoded, depth_encoded, K):
    # Decode base64 encoded color and depth image data
    color_image_data = base64.b64decode(color_encoded)
    depth_image_data = base64.b64decode(depth_encoded)
    print(f"len depth_image_data: {len(depth_image_data)}\n")
    
    # Load color image from bytes (assuming it's a PNG format in this example)
    try:
        color_image = cv2.imdecode(np.frombuffer(color_image_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Load depth image from bytes (assuming it's a PNG format in this example)
        depth_image = cv2.imdecode(np.frombuffer(depth_image_data, dtype=np.uint16), cv2.IMREAD_UNCHANGED)
    except ValueError as e:
        print(f"Error loading image: {e}   || len depth_image_data: {len(depth_image_data)}\n")
        return None
    # Create Open3D RGBD image from color and depth data
    depth_raw = o3d.geometry.Image(depth_image.astype(np.float32) / 1000.0)  # Assuming depth is in millimeters
    color_raw = o3d.geometry.Image(color_image)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    
    # Create Open3D point cloud from RGBD image with custom intrinsic matrix K
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    # intrinsics.set_intrinsics(color_image.shape[1], color_image.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    intrinsics.set_intrinsics(color_image.shape[1], color_image.shape[0], K[0][0], K[1][1], K[0][2], K[1][2])


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
    
    print(f"PCD len before downsampling {len(pcd.points)}")
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
    print(f"PCD len after downsampling {len(pcd.points)}")
    # pcd.estimate_normals()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    print(f"PCD created with {len(pcd.points)} points")
    return pcd



def process_frame(message_json_list, frame_id):
    pcd_list = []
    i=1
    for message in message_json_list:
        print(f"Processing message {i}")
        camera_name = message.get('camera_name')
        enc_c = message.get('enc_c')
        enc_d = message.get('enc_d')
        ts = message.get('send_ts')
        K = message.get('K')
        
        # create point cloud
        pc = create_pc_from_encoded_data(enc_c, enc_d, K)
        if pc is None:
            print(f"Error creating point cloud for camera {camera_name} with frame {frame_id}")
            i += 1
            continue
        # Append point cloud to frame point cloud list
        pcd_list.append(pc)
        i += 1
        # if len(pcd_list) > 2: # for testing
        #     break
        
    print(f"PCD List Length = {len(pcd_list)}")
    
    target_registration = message.get('target_model')
    registration_func = registration_methods.get(target_registration)
    if registration_func:
        final_fused_point_cloud = registration_func(pcd_list)
    else:
        raise ValueError(f"Unknown registration method: {target_registration}")
    
    
    # Save to Blob Storage
    # blob_name_reg = f"reg_{ts}_{frame_id}.ply"
    blob_name_reg = f"reg_{frame_id}.ply"
    save_and_upload_pcd(final_fused_point_cloud, blob_name_reg)

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    # print(f"Received message: {msg.topic} {msg.payload}")
    
    if True: #try:
        message = json.loads(msg.payload)
        frame_id = message.get('frame_id')
        print(f"Received MSG with FrameID = {frame_id}")
        if frame_id not in received_frames_dict:
            received_frames_dict[frame_id] = []
        
        received_frames_dict[frame_id].append(message)
        if len(received_frames_dict[frame_id]) == CAMERA_COUNT:
            print(f"received_frames_dict[frame_id] == CAMERA_COUNT GOING TO PROCESS {frame_id}")
            # Start a new process for the long time function
            frame_data_copy = copy.deepcopy(received_frames_dict[frame_id])
            process = Process(target=process_frame, args=(frame_data_copy, frame_id,))
            process.start()
            received_frames_dict[frame_id] = []
        else:
            print(f"Count = {len(received_frames_dict[frame_id])}")


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
    
    
main()