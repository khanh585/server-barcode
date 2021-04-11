import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import time

def uploadVideo(container_name, pathVideo, conn_str, filename):
    try:
        # Create a blob client using the local file name as the name for the blob
        container_client = ContainerClient.from_connection_string(conn_str=conn_str, container_name=container_name)
        blob_client = container_client.get_blob_client(filename)

        print("\nUploading to Azure Storage as blob:\n\t" + pathVideo)
        t0 = time.time()
        # Upload the created file
        url = ''
        with open(pathVideo, "rb") as data:
            blob_client.upload_blob(data)
            url = blob_client.url
        print(time.time() - t0)
        print('upload DONE!')
    except Exception as ex:
        print(time.time() - t0)
        print('Exception:')
        print(ex)
    finally:
        return url
