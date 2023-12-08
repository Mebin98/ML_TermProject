import sagemaker
import boto3
import os
import json
import numpy as np
import pandas as pd

# get current SageMaker session
sagemaker_session = sagemaker.Session()

# print(dir(sagemaker_session))
# print('sagemaker session : ', sagemaker_session)
# default_bucket_name = sagemaker_session.default_bucket()

bucket_name = 'sagemaker-gcu-spotify'


def get_s3_keys():
    
    """
    return keys in use for accessing s3 bucket 
    """
    
    aws_cli = 'aws configure get aws_access_key_id' 
    secret_key_cmd = 'aws configure get aws_secret_access_key'
    
    access_key = os.popen(aws_cli).read()
    secret_key = os.popen(secret_key_cmd).read()
    
    return access_key, secret_key


def authentification_process():
    
    access_key, secret_key = get_s3_keys()
    region = boto3.session.Session().region_name
    
    # s3_client = boto3.client('s3',
    #                             region_name=region, 
    #                             aws_access_key_id=access_key,
    #                             aws_secret_access_key=secret_key)

    s3_client = boto3.client('s3', region_name=region)
                                
    return s3_client

#s3 bucket name 

def download_from_bucket(resource_name:str, prefix:str='/data/', s3_mode:bool=True):
    """
    This function reads resource_name and returns the resource
    from the Amazon s3 bucket resource we've set up.
    
    Args:
    
    resource_name (str) : The name of the resource to download from.
    
    prefix (str) : The path to the file to download.(s3 bucket)
    
    s3_mode (bool) : use predefined s3 bucket if it has True; else use default_bucket
                    from the sagemaker session object.
    """
    
    
    bucket_name = 'sagemaker-gcu-spotify'
    
    bucket_name = bucket_name if s3_mode else sagemaker.Session().default_bucket() 
    local_path = '/home/ec2-user/environment/ml/'
    s3_client = authentification_process()
    
                                
    s3_client.download_file(bucket_name, prefix + resource_name, local_path + resource_name)

    

def upload_to_bucket(resource_name:str, prefix:str='data/', s3_mode:bool=True):
    """
    read resource_name and upload the resource
    to the Amazon s3 bucket resource we've set up.
    
    Args:
    
    resource_name (str) : The name of the resource to upload to.
    
    prefix (str) : The path of the s3 bucket where resource to be uploaded.(s3 bucket)
    
    s3_mode (bool) : use predefined s3 bucket if it has True; else use default_bucket
                    from the sagemaker session object.
    """
    
    
    bucket_name = 'sagemaker-gcu-spotify'
    
    bucket_name = bucket_name if s3_mode else sagemaker.Session().default_bucket() 
    local_path = '/home/ec2-user/environment/ml/'
    
    s3_client = authentification_process()
    
    response = s3_client.upload_file(local_path + resource_name, bucket_name, prefix + resource_name)
    print(response)

# upload_to_bucket('spotify_data.csv', prefix='data/')

download_from_bucket('spotify_data.csv')
