from .githubUtils import download_and_upload_to_s3, get_github_repo_stars
import datetime
import os 
from dotenv import load_dotenv
import boto3
from .githubUtils import get_github_repo_stars
import json
import aiohttp

scanner_path = os.path.dirname(__file__)
root_path = os.path.dirname(os.path.dirname(scanner_path))
comfy_path = os.path.join(root_path,'comfyui-fork')
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

def update_package_total_nodes(ddb_package_table, pacakge_id, total_nodes_value):
    if ddb_package_table is None or pacakge_id is None or total_nodes_value is None:
        print('🔴ddb_package_table is None')
        return None
    response = ddb_package_table.update_item(
        Key={
            'id': pacakge_id,
        },
        UpdateExpression='SET totalNodes = :val',
        ExpressionAttributeValues={
            ':val': total_nodes_value
        },
        ReturnValues="UPDATED_NEW"
    )

    print('ddb package item updated with totalNodes', '\n')
    return response

def create_pacakge_ddb(pacakge_data:dict):
    repo_url = pacakge_data['reference']
    title = pacakge_data['title']
    description = pacakge_data['description']
    repo_data = get_github_repo_stars(repo_url)
    owner_avatar_url= repo_data['owner_avatar_url'] if 'owner_avatar_url' in repo_data else None
    # image_url = repo_data['image_url'] if 'image_url' in repo_data else None
    star_count = repo_data['stars'] if 'stars' in repo_data else None
    try:
        repo_useranme = repo_url.split('/')[-2]
        repo_name = repo_url.split('/')[-1]
        # if get_package_ddb(repo_name) is not None:
        #     repo_name = repo_name + "_" + repo_useranme
        print("\n⭐️repo_name",repo_name)
        item = {
            'id': repo_name,  # Convert UUID to string
            "authorID": "admin",
            'gitHtmlUrl': repo_url,
            # 'gitUsername': repo_useranme,
            'title': title,
            'description': description,
            'ownerGitAvatarUrl': owner_avatar_url,
            # 'imageUrl': image_url,
            'totalStars': star_count,
            'updatedAt': datetime.datetime.now().replace(microsecond=0).isoformat()
        }
        print('ddb package item created','\n')
        response = ddb_package_table.put_item(Item=item)
        return item

    except Exception as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            print("A record with the same gitHtmlUrl was inserted concurrently.")
        print("Error adding package item to DynamoDB:", e)
        return None

#####v2######
def put_node_package_ddb(item):
    try:
        repo_data = get_github_repo_stars(item.get('gitHtmlUrl'))
        owner_avatar_url= repo_data['owner_avatar_url'] if 'owner_avatar_url' in repo_data else None
        star_count = repo_data['stars'] if 'stars' in repo_data else None
        
        webDir = item.get('webDir')
        jsFilePaths = None
        if webDir:
            jsFilePaths = json.dumps(download_and_upload_to_s3(item['gitRepo'], webDir))
        response = ddb_package_table.put_item(Item={
            **item,
            'updatedAt': datetime.datetime.now().replace(microsecond=0).isoformat(),
            'totalStars': star_count,
            'ownerGitAvatarUrl': owner_avatar_url,
            'description': repo_data.get('description',''),
            'jsFilePaths': jsFilePaths
        })
        return item
    except Exception as e:
        print("❌🔴Error adding package item to DynamoDB:", e)
        return None

def put_node_ddb(item):
    try:
        response = ddb_node_table.put_item(Item={
            **item,
            'updatedAt': datetime.datetime.now().replace(microsecond=0).isoformat(),
        })
        return item
    except Exception as e:
        print("❌🔴Error adding package item to DynamoDB:", e)
        return None

######v3

def custom_serializer(obj):
    """Convert non-serializable objects."""
    if isinstance(obj, (list, tuple, set)):
        return list(obj)  # Convert tuples and sets to lists
    elif isinstance(obj, dict):
        # Recursively apply to dictionary entries
        return {str(key): custom_serializer(value) for key, value in obj.items()}
    else:
        return obj

from .githubUtils import get_repo_user_and_name
from decimal import Decimal
import time
from  scanner.analyze_node_input import analyze_class

def write_to_db_record(input_dict): 
    # WRITE TO DDB
    NODE_CLASS_MAPPINGS = input_dict['NODE_CLASS_MAPPINGS']
    NODE_DISPLAY_NAME_MAPPINGS = input_dict['NODE_DISPLAY_NAME_MAPPINGS']
    cur_node_package = input_dict['cur_node_package']
    module_path = input_dict['module_path']
    prev_nodes = input_dict['prev_nodes']
    success = input_dict['success']
    time_before = input_dict['time_before']
    if 'ComfyUI-Manager' in module_path:
        return
    nodes_count = len(NODE_CLASS_MAPPINGS) - len(prev_nodes)
    import_time = time.perf_counter() - time_before
    
    username, repo_name, default_branch_name = get_repo_user_and_name(module_path)
    print('🍻 cur_node_package',cur_node_package)
    packageID = username + '_' + repo_name
    custom_node_defs = {}
    for name in NODE_CLASS_MAPPINGS:
        try:
            if name not in prev_nodes: 
                paths = analyze_class(NODE_CLASS_MAPPINGS[name])
                # all_node = fetch_node_info()
                node_def = node_info(input_dict, name)
                data = {
                    "id": name+"~"+packageID,
                    "nodeType": name, 
                    "nodeDef": json.dumps(node_def), 
                    "packageID": packageID,
                    "gitRepo": username + '/' + repo_name}
                custom_node_defs[name] = node_def
                if paths is not None and len(paths) > 0:
                    data['folderPaths'] = json.dumps(paths, default=custom_serializer)
                put_node_ddb(data)
        except Exception as e:
            print("❌analyze imported node: error",e)
    put_node_package_ddb({
        **cur_node_package,
        'id': packageID,
        'gitRepo': username + '/' + repo_name,
        'gitHtmlUrl': 'https://github.com/'+username + '/' + repo_name,
        'nameID': repo_name,
        'authorID': 'admin',
        'status': 'IMPORT_'+ ('SUCCESS' if success else 'FAILED'),
        'defaultBranch': default_branch_name,
        'totalNodes':nodes_count,
        "importTime": Decimal(str(import_time)),
        'nodeDefs': json.dumps(custom_node_defs)
    })

# For COMFYUI BASE NODES
def save_base_nodes_to_ddb(NODE_CLASS_MAPPINGS):
    baseNodeDefs = {}
    for name in NODE_CLASS_MAPPINGS: 
        paths = analyze_class(NODE_CLASS_MAPPINGS[name])
        node_def = node_info(name)
        data = {
            "id": name+"~"+'comfyanonymous_ComfyUI',
            "nodeType": name, 
            "nodeDef": json.dumps(node_def), 
            "packageID": 'comfyanonymous_ComfyUI',
            "gitRepo": 'comfyanonymous/ComfyUI'}
        baseNodeDefs[name] = node_def
        if paths is not None and len(paths) > 0:
            data['folderPaths'] = json.dumps(paths, default=custom_serializer)
        put_node_ddb(data)
    put_node_package_ddb({
                    'id': 'comfyanonymous_ComfyUI',
                    'gitRepo': "comfyanonymous/ComfyUI",
                    'gitHtmlUrl': 'https://github.com/comfyanonymous/ComfyUI',
                    'nameID': 'ComfyUI',
                    'authorID': 'admin',
                    'status': 'IMPORT_SUCCESS',
                    'defaultBranch': 'master',
                    'totalNodes':len(NODE_CLASS_MAPPINGS),
                    'nodeDefs': json.dumps(baseNodeDefs)
                })

# copied from server.py
def node_info(input_dict, node_class:str):
    NODE_CLASS_MAPPINGS = input_dict['NODE_CLASS_MAPPINGS']
    NODE_DISPLAY_NAME_MAPPINGS = input_dict['NODE_DISPLAY_NAME_MAPPINGS']
    obj_class = NODE_CLASS_MAPPINGS[node_class]
    info = {}
    info['input'] = obj_class.INPUT_TYPES()
    info['output'] = obj_class.RETURN_TYPES
    info['output_is_list'] = obj_class.OUTPUT_IS_LIST if hasattr(obj_class, 'OUTPUT_IS_LIST') else [False] * len(obj_class.RETURN_TYPES)
    info['output_name'] = obj_class.RETURN_NAMES if hasattr(obj_class, 'RETURN_NAMES') else info['output']
    info['name'] = node_class
    info['display_name'] = NODE_DISPLAY_NAME_MAPPINGS[node_class] if node_class in NODE_DISPLAY_NAME_MAPPINGS.keys() else node_class
    info['description'] = obj_class.DESCRIPTION if hasattr(obj_class,'DESCRIPTION') else ''
    info['category'] = 'sd'
    if hasattr(obj_class, 'OUTPUT_NODE') and obj_class.OUTPUT_NODE == True:
        info['output_node'] = True
    else:
        info['output_node'] = False

    if hasattr(obj_class, 'CATEGORY'):
        info['category'] = obj_class.CATEGORY
    return info


import json
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

def fetch_node_info():
    url = "http://localhost:8188/object_info"
    
    try:
        with urlopen(url) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                return data
            else:
                print(f"Failed to fetch node info. Status code: {response.status}")
                return None
    except HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
        return None
    except URLError as e:
        print(f"URL Error: {e.reason}")
        return None
