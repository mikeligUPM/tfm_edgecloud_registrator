# MQTT Broker settings
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
MQTT_TOPIC = 'cameraframes'

# Azure Blob Storage settings
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stgpointcloud;AccountKey=bLVqoP9n84liv8b+nQm2G+v4Jj+VLvetmViC1xWuHId428mxL61O5Sg9zz8jbxsLaFpBvDhan0Oi+AStGz//zg==;EndpointSuffix=core.windows.net"
BLOB_CONTAINER_NAME = "cameraframes"
CAMERA_COUNT = 8

VOXEL_SIZE = 0.00001
THRESHOLD = 0.001 * VOXEL_SIZE