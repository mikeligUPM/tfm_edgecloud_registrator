# import azure.functions as func
# import datetime
# import json
# import logging

# app = func.FunctionApp()


# @app.blob_trigger(arg_name="myblob", path="cameraframes",
#                                connection="AzureWebJobsStorage") 
# def BlobTrigger(myblob: func.InputStream):
#     logging.info(f"Python blob trigger function processed blob"
#                 f"Name: {myblob.name}"
#                 f"Blob Size: {myblob.length} bytes")


from azure.storage.blob import BlobServiceClient
import azure.functions as func
import logging

app = func.FunctionApp()

BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stgpointcloud;AccountKey=bLVqoP9n84liv8b+nQm2G+v4Jj+VLvetmViC1xWuHId428mxL61O5Sg9zz8jbxsLaFpBvDhan0Oi+AStGz//zg==;EndpointSuffix=core.windows.net"
BLOB_CONTAINER_NAME = "cameraframes"
MAX_BLOB_NUM = 30


def get_blob_service_client_connection_string():
    connection_string = BLOB_CONNECTION_STRING
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    return blob_service_client

def list_blobs_sorted(blob_service_client: BlobServiceClient, container_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = list(container_client.list_blobs())

    # Sort blobs by last modified time (if available) or use creation time
    sorted_blobs = sorted(blob_list, key=lambda b: b.creation_time, reverse=True)
    # logging.info(f"Sorted blobs: {sorted_blobs}")
    return sorted_blobs

def delete_old_blobs(blob_service_client: BlobServiceClient, container_name, num_to_keep=MAX_BLOB_NUM):
    sorted_blobs = list_blobs_sorted(blob_service_client, container_name)

    if len(sorted_blobs) > num_to_keep:
        blobs_to_delete = sorted_blobs[num_to_keep:]

        for blob in blobs_to_delete:
            container_client = blob_service_client.get_container_client(container_name)
            container_client.delete_blob(blob.name)
            logging.info(f"Deleted blob: {blob.name}")
    else:
        logging.info(f"Blob count {len(sorted_blobs)} is less than {num_to_keep}. No blobs to delete.")


@app.blob_trigger(arg_name="myblob", path=BLOB_CONTAINER_NAME,
                               connection="AzureWebJobsStorage") 
def blob_trigger_py(myblob: func.InputStream):
    logging.info(f"Blob trigger function processed blob\n"
                 f"Name: {myblob.name}\n"
                 f"Blob Size: {myblob.length} bytes")

    blob_service_client = get_blob_service_client_connection_string()
    if blob_service_client:
        delete_old_blobs(blob_service_client, BLOB_CONTAINER_NAME)
    else:
        logging.error("Failed to get BlobServiceClient")