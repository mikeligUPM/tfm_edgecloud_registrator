import numpy as np

# MQTT Broker settings
MQTT_BROKER = 'localhost'
MQTT_PORT = 1883
MQTT_TOPIC = 'cameraframes'

# Azure Blob Storage settings
BLOB_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stgpointcloud;AccountKey=bLVqoP9n84liv8b+nQm2G+v4Jj+VLvetmViC1xWuHId428mxL61O5Sg9zz8jbxsLaFpBvDhan0Oi+AStGz//zg==;EndpointSuffix=core.windows.net"
BLOB_CONTAINER_NAME = "cameraframes"
CAMERA_COUNT = 8



# VOXEL_SIZE = 0.00001
# THRESHOLD = 0.01 * VOXEL_SIZE

# BACKBONE_INIT_VOXEL_SIZE = 0.00001 * 2.5# BAD RESULT
# # BACKBONE_INIT_VOXEL_SIZE = 0.00025
# # BACKBONE_INIT_VOXEL_SIZE = 0.0001 * 2.5  * 2 # 5, 10 runtime

# BACKBONE_INIT_RADIUS = BACKBONE_INIT_VOXEL_SIZE * 20
# BACKBONE_INIT_RADIUS = BACKBONE_INIT_VOXEL_SIZE * 2


# BACKBONE_INIT_VOXEL_SIZE = VOXEL_SIZE = 0.0001 * 2.5
# BACKBONE_INIT_VOXEL_SIZE = VOXEL_SIZE = 0.025
# # BACKBONE_INIT_VOXEL_SIZE = VOXEL_SIZE = 0.00001 * 2.5
# BACKBONE_INIT_RADIUS = 2.5 #/ 100
# # BACKBONE_INIT_RADIUS = BACKBONE_INIT_VOXEL_SIZE * 2
# #threshold = 0.00025  * 10 #good results in calculate metrics with own data
# threshold = 0.01




        