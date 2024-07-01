from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import open3d as o3d
import os

from constants import BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

if not container_client.exists():
    container_client.create_container()
    

def save_and_upload_pcd(pcd, blob_name):
    save_point_cloud_to_ply(pcd, blob_name)
    if upload_ply_to_blob_storage(blob_name):
        os.remove(blob_name)
        print(f"{blob_name} removed from disk.")
    
def save_point_cloud_to_ply(point_cloud, file_path):
    try:
        o3d.io.write_point_cloud(file_path, point_cloud)
        print(f"Point cloud saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving point cloud to PLY: {e}")
        print(f"point_cloud class == {point_cloud.__class__}  file_path == {file_path.__class__}")
        return False

def upload_ply_to_blob_storage(blob_name):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)
        
        with open(blob_name, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        print(f"{blob_name} uploaded to Azure Blob Storage.")
        return True
    except Exception as e:
        print(f"Error uploading file to Blob Storage: {e}")
        return False