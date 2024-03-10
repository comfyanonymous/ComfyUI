from .githubUtils import download_and_upload_to_s3, get_github_repo_stars
import datetime
import os 
from dotenv import load_dotenv
import boto3
from .githubUtils import get_github_repo_stars
import json

scanner_path = os.path.dirname(__file__)
root_path = os.path.dirname(os.path.dirname(scanner_path))
comfy_path = os.path.join(root_path,'comfyui-fork')
print('root path',root_path)
load_dotenv(os.path.join(root_path, '.env.local'))
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
node_table_name = "ComfyNode" + os.getenv('DDB_TABLE_POSTFIX')# DDB_TABLE_CUSTOM_NODE
package_table_name = "ComfyNodePackage" + os.getenv('DDB_TABLE_POSTFIX')
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