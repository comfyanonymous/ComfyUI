import requests

def get_github_repo_stars(repo_url):
    # Extracting the owner and repo name from the URL
    parts = repo_url.split("/")
    owner, repo = parts[-2], parts[-1]

    # GitHub API endpoint for fetching repo details
    api_url = f"https://api.github.com/repos/{owner}/{repo}"

    # Making the request
    response = requests.get(api_url)
    
    if response.status_code == 200:
        # Extracting the number of stars
        repo_data = response.json()
        stars = repo_data.get("stargazers_count", 0)
        owner_avatar_url = repo_data['owner']['avatar_url']
        image_url = get_first_image_url_from_readme(owner, repo)
        return {
            "stars": stars,
            "owner_avatar_url": owner_avatar_url,
            "image_url": image_url
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
