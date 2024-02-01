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
from githubUtils import get_github_repo_stars
from manager_copy import gitclone_install

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
node_table_name = os.getenv('DDB_TABLE_NODE')# DDB_TABLE_CUSTOM_NODE
package_table_name = os.getenv('DDB_TABLE_PACKAGE')
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

def gitclone_install2222(repo_url: str, target_dir: str):
    print('gitclone_install',repo_url, target_dir)
    # Save the current working directory
    # original_cwd = os.getcwd()

    try:
        # Change the current working directory to custom_node_path
        # os.chdir(custom_node_path)

        # Running the git clone command
        process = subprocess.Popen(['git', 'clone', repo_url, target_dir], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        # Reading and printing the output
        for line in iter(process.stdout.readline, ''):
            print(line, end='')

        # Wait for the process to complete and get the exit code
        process.wait()
        print("📟 installed repo", repo_url)
        return process.returncode == 0

    except Exception as e:
        print(f"An error occurred: {e}")
        return False

    # finally:
        # Change back to the original directory
        # os.chdir(original_cwd)

def run_main_py_and_wait(package_data:dict):
    if os.path.exists(communication_file):
        print("Removing communication file")
        os.remove(communication_file)
    os.chdir(comfy_path)
    process = subprocess.Popen(['python', 'main.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output.startswith("__workspace_scanner__"):
                print(f"\033[93m{output}\033[0m", end='')  # end='' to avoid double newline
            else:
                print(output, end='')
    finally:
        process.terminate()
        process.wait()
        print(f"\033[93m Done importing:{package_data['reference']}\033[0m", end='')  
        # Create package and node in DynamoDB
        package = create_pacakge_ddb(package_data)
        print(f"📦package: {package['title']}")
        with open(communication_file, 'r') as file:
             for line in file:
                try:
                    data = json.loads(line.strip())
                    if 'node_type' in data:
                        print(f"💡node: {data['node_type']}")
                        create_node_dydb({
                            'nodeType': data['node_type'],  
                            'nodeDef': data['node_def'],
                            'folderPaths': data['folder_paths'],
                            'gitHtmlUrl': package_data['reference'],
                            'packageID': package['id']
                        })
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
        
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
            'createdAt': datetime.datetime.now().replace(microsecond=0).isoformat()
        }
        print('ddb package item created','\n')
        response = ddb_package_table.put_item(Item=item)
        return item

    except Exception as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            print("A record with the same gitHtmlUrl was inserted concurrently.")
        print("Error adding package item to DynamoDB:", e)
        return None
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
        existing = get_node_ddb(node_type)
        if existing is not None and existing['packageID'] != package_id:
            node_type = node_type + "_" + package_id
    except Exception as e:
        print("Error getting node item from DynamoDB:", e)
    try:
        item = {
            'id': node_type.replace(' ', '_'), 
            "authorID": "admin",
            'nodeType': node_type,
            "nodeDef": json.dumps(node_def),
            "folderPaths": json.dumps(data['folderPaths']),
            "packageID": package_id,
            'gitHtmlUrl': repo_url,
        }
        response = ddb_node_table.put_item(Item=item)
        print("👌ddb node item added:", item)

    except Exception as e:
        print("Error adding node item to DynamoDB:", e)

def delete_installed_node(target_dir: str):
    # Delete the installed node
    shutil.rmtree(target_dir)

def process_json(file_path):
    if (os.path.exists(file_path) == False):
        print("🔴file not found", file_path)
        gitclone_install("https://github.com/ltdrdata/ComfyUI-Manager", os.path.join(custom_node_path, "ComfyUI-Manager"))
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            for index, node in enumerate(data["custom_nodes"][15:16]):
                print(f"🗂️🗂️No.{index} files", node['files'])
                repo = node['reference']
                if 'github' not in repo:
                    continue
                if repo.endswith('.git'):
                    repo = repo[:-4]
                if repo.endswith('/'):
                    repo = repo[:-1]
                repo_name = repo.split('/')[-1]
                print('repo name',repo_name)
                target_dir = os.path.join(custom_node_path, repo_name)
                git_clone_url = repo + '.git' if not repo.endswith('.git') else repo
                # gitclone_install( git_clone_url, target_dir)
                gitclone_install([git_clone_url])
                run_main_py_and_wait({
                    'reference': repo,
                    'title': node['title'],
                    'description': node['description'],
                    'author': node['author'],
                })
                delete_installed_node(target_dir)
    except Exception as e:
        return f"An error occurred: {e}"


process_json(os.path.join(manager_path, "custom-node-list.json"))
