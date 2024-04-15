import os

from google.cloud import storage


def extract_bucket_name_and_path(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("The uri must follow gs://{bucket-name}/{path}")
    try:
        bucket_name, path = uri.split("gs://")[1].split("/", 1)
    except ValueError:
        raise ValueError("The uri must follow gs://{bucket-name}/{path}")
    return bucket_name, path


def get_bucket_and_path(gcs_client: storage.Client, uri: str) -> (storage.Bucket, str, bool):
    """
    Get the bucket and path from the uri.
    :return: The bucket, path and a boolean indicating which is true if the path is a file
    """
    bucket_name, path = extract_bucket_name_and_path(uri)
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(path)
    if blob.exists():
        return bucket, path, True

    prefix = path if path.endswith("/") else path + "/"
    iterator = gcs_client.list_blobs(bucket, prefix=prefix, delimiter="/")
    blobs = list(iterator)
    prefixes = list(iterator.prefixes)

    if blobs or prefixes:
        return bucket, path, False

    raise RuntimeError(f"The uri '{uri}' does not exist")


def get_bucket_and_file_path(gcs_client: storage.Client, uri: str):
    bucket, file_path, is_file = get_bucket_and_path(gcs_client, uri)
    if is_file:
        return bucket, file_path
    else:
        raise ValueError(f"The uri '{uri}' is a directory, not a file")

def download_gcs_file(gcs_client: storage.Client, uri: str, target_file_path: str):
    bucket, file_path = get_bucket_and_file_path(gcs_client, uri)
    blob = bucket.blob(file_path)
    blob.download_to_filename(target_file_path)

    if not os.path.exists(target_file_path):
        raise ValueError(f"File not downloaded to {target_file_path}")