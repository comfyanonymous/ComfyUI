import datetime
import json
import os
import shutil
import subprocess
import os
import subprocess
import boto3
from dotenv import load_dotenv
import os
import uuid
from botocore.exceptions import BotoCoreError, ClientError
from boto3.dynamodb.conditions import Key
from .manager_copy import run_script

scanner_path = os.path.dirname(__file__)
root_path = os.path.dirname(os.path.dirname(scanner_path))
comfy_path = os.path.join(root_path,'comfyui-fork')
communication_file = os.path.join(comfy_path, 'communication_file.txt') 
custom_node_path = os.path.join(comfy_path, "custom_nodes")
manager_path = os.path.join(comfy_path, "custom_nodes", "ComfyUI-Manager")
print('root path',root_path)
load_dotenv(os.path.join(root_path, '.env.local'))
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
node_table_name = "Node" + os.getenv('DDB_TABLE_POSTFIX')# DDB_TABLE_CUSTOM_NODE
package_table_name = "NodePackage" + os.getenv('DDB_TABLE_POSTFIX')
log_file = os.path.join(root_path, 'log.txt')

if not aws_access_key_id or not aws_secret_access_key:
    print("!!!!Missing AWS credentials")
    raise ValueError("Missing AWS credentials")

# Initialize a DynamoDB client
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name='us-west-1'  # Specify your AWS region
)
ddb_node_table = dynamodb.Table(node_table_name)
ddb_package_table = dynamodb.Table(package_table_name)

cur_git_repo = None
def run_main_py_and_wait(package_data:dict,index: int = 0):
    global cur_git_repo
    # print(f"😻git clone installing:{package_data['reference']}")
    if os.path.exists(communication_file):
        print("Removing communication file")
        os.remove(communication_file)
    os.chdir(comfy_path)
    try:
        cur_git_repo=package_data['title']
        cmd = ['python', 'main.py']  # Use 'python3' instead of 'python' if your system requires it
        # You can specify the current working directory (cwd) if needed, or use '.' for the current directory
        run_script(cmd, cwd='.')
    
    finally:
        # process.terminate()
        # process.wait()
        print(f"\033[93m Done importing:{package_data['reference']}\033[0m", end='')  
        # Create package and node in DynamoDB
        # package = create_pacakge_ddb(package_data)
        # print(f"📦created pacakge ddb: {package['title']}, index: {index}")
        # totalCount = 0 
        # if not os.path.exists(communication_file):
        #     print("🔴communication_file not found")
        # with open(communication_file, 'r') as file:
        #      for line in file:
        #         try:
        #             data = json.loads(line.strip())
        #             if 'node_type' in data:
        #                 print(f"💡node: {data['node_type']}")
        #                 create_node_dydb({
        #                     'nodeType': data['node_type'],  
        #                     'nodeDef': data['node_def'],
        #                     'folderPaths': data['folder_paths'],
        #                     'gitHtmlUrl': package_data['reference'],
        #                     'packageID': package['id']
        #                 })
        #                 totalCount += 1
        #         except json.JSONDecodeError as e:
        #             print(f"Error decoding JSON: {e}")
        
def get_package_ddb(id:str):
    response = ddb_package_table.get_item(
        Key={
            'id': id,
        })
    if 'Item' in response:
        return response['Item']
    else:
        return None
def packageGitUrlExists(url:str):
    response = ddb_package_table.query(
        IndexName='gitHtmlUrl',
        KeyConditionExpression=Key('gitHtmlUrl').eq(url)
    )
    if response['Items']:
        # gitHtmlUrl already exists, handle accordingly
        print("A record with the same gitHtmlUrl already exists.")
        return True
    return False

def get_node_ddb(id:str):
    try:
        response = ddb_node_table.get_item(
            Key={
                'id': id,
            })
        if 'Item' in response:
            return response['Item']
        else:
            return None
    except Exception as e:
        print("Error getting node item from DynamoDB:", e)
def create_node_dydb(data):
    print('!!!!create_node_dydb',data)
    try:
        node_type = data['nodeType']
        node_def = data['nodeDef']
        repo_url = data['gitHtmlUrl']
        package_id = data['packageID']
        node_id = node_type.replace(' ', '_')
        existing = get_node_ddb(node_id)
        isDuplicateName = False
        if existing is not None and existing['packageID'] != package_id:
            node_id = node_id + "_" + package_id
            if node_type == existing['nodeType']:
                isDuplicateName = True
    except Exception as e:
        print("Error getting node item from DynamoDB:", e)
    try:
        item = {
            'id': node_id, 
            "authorID": "admin",
            'nodeType': node_type,
            "nodeDef": json.dumps(node_def),
            "folderPaths": json.dumps(data['folderPaths']),
            "packageID": package_id,
            'gitHtmlUrl': repo_url,
            "isDuplicateName": isDuplicateName,
            'updatedAt': datetime.datetime.now().replace(microsecond=0).isoformat()
        }
        response = ddb_node_table.put_item(Item=item)
        print("👌ddb node item added id:", item.id)

    except Exception as e:
        print("Error adding node item to DynamoDB:", e)

def delete_installed_node(target_dir: str):
    # Delete the installed node
    shutil.rmtree(target_dir)
