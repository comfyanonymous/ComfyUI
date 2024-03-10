import mimetypes
import subprocess
import requests
import os
import shutil
from dotenv import load_dotenv
from dotenv import load_dotenv
import boto3

scanner_path = os.path.dirname(__file__)
root_path = os.path.dirname(os.path.dirname(scanner_path))
load_dotenv(os.path.join(root_path, '.env.local'))
github_key = os.getenv('GITHUB_API_KEY')
s3_bucket_name = os.getenv('S3_BUCKET_NAME')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')

def get_github_repo_stars(repo_url):
    if github_key is None:
        print('🔴🔴 no github api key provided!!')
    # Extracting the owner and repo name from the URL
    parts = repo_url.split("/")
    owner, repo = parts[-2], parts[-1]

    # GitHub API endpoint for fetching repo details
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    
    headers = {
        "Authorization": f"token {github_key}",  # Adding the authentication token to the request header
        "Accept": "application/vnd.github.v3+json"
    }

    # Making the authenticated request
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        # Extracting the number of stars
        repo_data = response.json()
        stars = repo_data.get("stargazers_count", 0)
        owner_avatar_url = repo_data['owner']['avatar_url']
        # image_url = get_first_image_url_from_readme(owner, repo)
        description = repo_data.get('description')
        if not description:
            # Fetch README content
            readme_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
            readme_info = requests.get(readme_url, headers=headers).json()
            readme_content = readme_info.get('content', '')
            readme_text = requests.get(readme_info['download_url']).text if 'download_url' in readme_info else ""
            paragraphs = readme_text.split('\n\n')
            # Keep only the first two paragraphs
            first_two_paragraphs = paragraphs[:2]
            description = '\n'.join(first_two_paragraphs)

        return {
            "stars": stars,
            "owner_avatar_url": owner_avatar_url,
            'description': description,
            # "image_url": image_url
        }
    else:
        print(f"Failed to retrieve repository data. Status Code: {response.status_code}")
        return {}
import base64
import re

def get_first_image_url_from_readme(owner, repo):
    # Construct the URL for the README API
    readme_api_url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    
    # Make a request to get the README
    readme_response = requests.get(readme_api_url)
    if readme_response.status_code == 200:
        readme_data = readme_response.json()
        # Decode the content from base64
        readme_content = base64.b64decode(readme_data['content']).decode('utf-8')
        
        # Use regular expressions to find image URLs in markdown and HTML <img> tags
        markdown_image_urls = re.findall(r'!\[.*?\]\((.*?)\)', readme_content)
        html_img_urls = re.findall(r'<img.*?src="(.*?\.(jpg|jpeg|png|gif|bmp|svg))".*?>', readme_content)
        
        # Combine the lists, maintaining order and removing duplicates
        all_image_urls = list(dict.fromkeys(markdown_image_urls + html_img_urls))
        
        if all_image_urls:
            return all_image_urls[0]  # Return the first image URL
        else:
            return "No image found in README"
    else:
        return "Error: Unable to fetch README information"

def clear_except_allowed_folder(path, allowedFolder):
    """
    Clears everything in the specified path except for the allowedFolder.
    
    :param path: Path to the directory to clear.
    :param allowedFolder: The name of the folder to keep.
    """
    # Make sure the path is a directory
    if not os.path.isdir(path):
        print(f"The provided path {path} is not a directory.")
        return

    # Iterate through items in the directory
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        # Check if the current item is the allowedFolder
        if item == allowedFolder:
            continue  # Skip the allowedFolder
        
        # If item is a directory, remove it and its contents
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
            print(f"Removed directory: {item_path}")
        # If item is a file, remove it
        else:
            os.remove(item_path)
            print(f"Removed file: {item_path}")

######v2
def get_repo_user_and_name(module_path):
    command = ['git', 'config', '--get', 'remote.origin.url']
    try:
        result = subprocess.run(command, cwd=module_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        repo_url = result.stdout.strip()

        # Match both HTTPS and SSH URLs
        match = re.search(r'(?:https?://github.com/|git@github.com:)([^/]+)/([^/.]+)', repo_url)
        if match:
            username, repo_name = match.groups()
            # Attempt to get the default branch name
            branch_command = ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            branch_result = subprocess.run(branch_command, cwd=module_path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            main_branch = branch_result.stdout.strip()
            return username, repo_name, main_branch
        else:
            return "Could not parse URL", ""
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}", ""


s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)
def download_and_upload_to_s3(repo_url, webDir, current_path='', file_paths=None):
    """
    Recursively downloads contents from a specified directory in a GitHub repo and uploads them to S3,
    maintaining the directory structure. Collects paths of all files processed.
    
    :param repo_url: URL to the GitHub repository
    :param webDir: Path to the directory in the repository from which to start the download
    :param current_path: Keeps track of the current path for recursive calls
    :param file_paths: Accumulator for collecting file paths
    :return: List of all file paths processed
    """
    if file_paths is None:
        file_paths = []
    
    parts = repo_url.split("/")
    owner, repo = parts[-2], parts[-1]

    headers = {
        "Authorization": f"token {github_key}",
        "Accept": "application/vnd.github.v3+json"
    }
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{webDir}/{current_path}"

    response = requests.get(api_url, headers=headers)
    items = response.json()

    if not isinstance(items, list):  # If the response is not a list, it might be an error message
        print(f"Error fetching {api_url}: {items.get('message', 'Unknown error')}")
        return file_paths

    for item in items:
        if item['type'] == 'file':
            # Download the file content
            download_response = requests.get(item['download_url'])
            file_name = os.path.basename(item['path'])
            s3_key = f"packageWebDir/{owner}_{repo}/{current_path}{file_name}"
            
            # Collect file path
            file_paths.append(f"{current_path}{file_name}")
            mime_type = guess_mime_type(file_name)
            print(f"📄 Uploading typ {mime_type}, {file_name} to S3 with key {s3_key}")
            # Upload to S3
            s3_client.put_object(Bucket="comfyspace", Key=s3_key, Body=download_response.content, ContentType=mime_type)
            print(f"✅ Uploaded {file_name} to S3 with key {s3_key}")
        elif item['type'] == 'dir':
            # Recursively process the directory
            new_path = os.path.join(current_path, os.path.basename(item['path'])) + '/'
            print(f"📁 Processing directory: {new_path}")
            download_and_upload_to_s3(repo_url, webDir, new_path, file_paths)

    return file_paths

def guess_mime_type(file_name):
    mime_type, _ = mimetypes.guess_type(file_name)
    return mime_type or 'application/octet-stream'