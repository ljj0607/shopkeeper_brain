from minio import Minio
import os
import logging
from dotenv import load_dotenv
load_dotenv()

def get_minio_client():
    try:
        client = Minio(os.getenv("MINIO_ENDPOINT"),
                       access_key=os.getenv("MINIO_ACCESS_KEY"),
                       secret_key=os.getenv("MINIO_SECRET_KEY"),
                       secure=False
                       )

        bucket_name = os.getenv("MINIO_BUCKET_NAME")

        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logging.info(f"Created bucket {bucket_name}")
        else:
            logging.info(f"Bucket {bucket_name} already exists")
    except Exception as e:
        logging.error(e)
        return None
    return client
