import webbrowser

from pydrive.drive import GoogleDrive 
from pydrive.auth import GoogleAuth 

gauth = GoogleAuth()
gauth.LoadCredentialsFile("mycreds.txt")
if gauth.credentials is None:
    gauth.LocalWebserverAuth()
elif gauth.access_token_expired:
    gauth.Refresh()
else:
    gauth.Authorize()
gauth.SaveCredentialsFile("mycreds.txt")
  
drive = GoogleDrive(gauth)

notebook_name = "colab_runner.ipynb"

# Delete the files with the same name (clear the cache)
file_list = drive.ListFile({'q': f"title='{notebook_name}'"}).GetList()
if file_list:
    for file in file_list:
        file.Trash()
        file.UnTrash()
        file.Delete()

# Create the new updated file and run it
notebook_path = "./{}".format(notebook_name)
notebook_file = drive.CreateFile({ 'title': notebook_name })
notebook_file.SetContentFile(notebook_path)
notebook_file.Upload()
notebook_file = None

# Print all the existing files and go to the last one on 
# the list, hoping it is the most recent one
file_list = drive.ListFile({'q': "title='{}'".format(notebook_name)}).GetList()
if file_list:
    for file in file_list:
        print("Found file: {0} with ID: {1}".format(file['title'], file['id']))
    
    file_id = file_list[-1]['id']
    colab_url = "https://colab.research.google.com/drive/{}".format(file_id)
    print("Opening notebook: {}".format(colab_url))
    if webbrowser.open(colab_url):
        print("Check your browser!")
    else:
        print("Couldn't open the browser!")