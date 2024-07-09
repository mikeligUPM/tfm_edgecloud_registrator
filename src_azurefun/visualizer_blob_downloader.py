from azure.storage.blob import BlobServiceClient
import azure.functions as func
import logging


BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stgpointcloud;AccountKey=bLVqoP9n84liv8b+nQm2G+v4Jj+VLvetmViC1xWuHId428mxL61O5Sg9zz8jbxsLaFpBvDhan0Oi+AStGz//zg==;EndpointSuffix=core.windows.net"
BLOB_CONTAINER_NAME = "cameraframes"


def get_blob_service_client_connection_string():
    connection_string = BLOB_CONNECTION_STRING
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return blob_service_client

def list_blobs_sorted(blob_service_client: BlobServiceClient, container_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = list(container_client.list_blobs())

    blob_names = [blob.name for blob in blob_list]
    # # Sort blobs by last modified time (if available) or use creation time
    # sorted_blobs = sorted(blob_list, key=lambda b: b.creation_time, reverse=True)
    # # logging.info(f"Sorted blobs: {sorted_blobs}")
    return blob_names

def find_target_blob(frame_id, registration_method, dataset_name=None) -> str:
    blob_service_client = get_blob_service_client_connection_string()
    if blob_service_client:
        blob_list = list_blobs_sorted(blob_service_client, BLOB_CONTAINER_NAME)
        target_blob = f"reg__{frame_id}__{registration_method}.ply"
        if target_blob in blob_list:
            logging.info(f"Found target blob: {target_blob}. Going to download it.")
            blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=target_blob)
            with open(target_blob, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            logging.info(f"Downloaded blob: {target_blob}")
            return target_blob
    else:
        logging.error("Failed to get BlobServiceClient") 
        
def visualize():
    pass