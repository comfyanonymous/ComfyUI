

import os

import minio

from config.config import CONFIG
from framework.app_log import AppLog
import folder_paths
from PIL import Image


class MinIOConnection:
    """
    Singleton class.
    Manage the connection to minIO server.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            minio_setting = CONFIG['minio_settings']
            cls._instance.endpoint = minio_setting['endpoint']
            cls._instance.access_key = minio_setting['access_key']
            cls._instance.secret_key = minio_setting['secret_key']
            if 'region' in minio_setting and minio_setting['region']:
                cls._instance.region = minio_setting['region']
            else:
                cls._instance.region = None


            cls._instance.connection = minio.Minio(
                endpoint=cls._instance.endpoint,
                access_key=cls._instance.access_key,
                secret_key=cls._instance.secret_key,
                secure=True,
                region=cls._instance.region
            )
        return cls._instance


    # def __del__(self):
    #     print('close connection.')
    #     self.connection.close()


    def get_connection(self):
        return self.connection
    
    
    def get_default_bucket(self):
        return CONFIG["minio_settings"]["bucket"]
    
    
    def fget_object(self, obj_name, file_path=None, bucket=None):
        if bucket is None:
            bucket=self.get_default_bucket()
            
        if file_path is None:
            file_basename = os.path.basename(obj_name)
            file_dir = CONFIG["resource"]["in_img_path_local"]
            file_path = f"{file_dir}/{file_basename}"
            print(f"get object filepath: {file_path}")
            
        try:
            self.connection.fget_object(bucket_name=bucket,
                    object_name=obj_name,
                    file_path=file_path
                    )
            return file_path
        except Exception as e:
            AppLog.error(f"[ObjectStorage] fget_object, fail to get: {obj_name}")
            return None
        
        
    def fput_object(self, obj_name, file_path, bucket=None):
        if bucket is None:
            bucket=self.get_default_bucket()
            
        try:
            self.connection.fput_object(bucket_name=bucket,
                    object_name=obj_name,
                    file_path=file_path
                    )
        except Exception as e:
            AppLog.error(f"[ObjectStorage] fput_object, fail to put: {obj_name}")
            
        
    def get_object(self, obj_name, bucket=None):
        if bucket is None:
            bucket=self.get_default_bucket()
            
        try:
            response = self.connection.get_object(bucket_name=bucket,
                    object_name=obj_name
                    )
            if response.status == 200:
                data = response.data
                return data
            else:
                AppLog.warning(f"[ObjectStorage] get_object, fail to get: {obj_name}")
        except Exception as e:
            AppLog.error(f"[ObjectStorage] get_object, fail to get: {obj_name}")
        
        return None
    
    
    def put_object(self, obj_name, data, data_len, bucket=None):
        if bucket is None:
            bucket=self.get_default_bucket()
            
        try:
            self.connection.put_object(bucket_name=bucket,
                    object_name=obj_name,
                    data=data,
                    length = data_len
                    )
        except Exception as e:
            AppLog.error(f"[ObjectStorage] fput_object, fail to put: {obj_name}")
    
    
    
    def exist_object(self, obj_name, bucket=None):
        try:
            response = self.connection.stat_object(bucket, obj_name)
            return True
        except Exception as e:
            return False
    
    
class ResourceMgr:
    
    instance = None
    
    @staticmethod
    def register(mgr_instance):
        if isinstance(mgr_instance, ResourceMgrLocal) or isinstance(mgr_instance, ResourceMgrRemote):
            ResourceMgr.instance = mgr_instance
            return True
        else:
            return False




class ResourceMgrLocal(ResourceMgr):
    
    def __init__(self) -> None:
        super().__init__()
        ResourceMgr.register(self)
    
    def get_image(self, image_path, open=True):
        image_path = folder_paths.get_annotated_filepath(image_path)
        if open:
            img = Image.open(image_path)
        else:
            img = None    
        return image_path, img
    
    
    def exist_image(self, image_path):
        return folder_paths.exists_annotated_filepath(image_path)
    
    
    
    def after_save_image_to_local(self, local_path):
        return  local_path  
    
    
    
class ResourceMgrRemote(ResourceMgr):
    
    def __init__(self) -> None:
        super().__init__()
        ResourceMgr.register(self)
    
    
    def get_image(self, image_path, open=True):
        local_path = folder_paths.input_path_remote_to_local(image_path)
        image_path = MinIOConnection().fget_object(image_path, local_path)
        if open:
            img = Image.open(image_path)
        else:
            img = None
        return image_path, img
    
    
    def exist_image(self, image_path):
        return MinIOConnection().exist_object(image_path)
    
    
    
    def after_save_image_to_local(self, local_path):
        file_basename = os.path.basename(local_path)
        remote_dir = CONFIG["resource"]["out_img_path_cloud"]
        remote_path = f"{remote_dir}/{file_basename}"
        MinIOConnection().fput_object(remote_path, local_path)
        
        AppLog.info(f"[ResMgr] after_save_image_to_local, remote path: {remote_path}")
        return remote_path