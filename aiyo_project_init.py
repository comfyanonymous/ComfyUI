
import json, os
import time
import urllib.request
import shutil
import traceback

from git import repo
import subprocess


def ab_download_resource(url, file_path, cache_resource=False, cache_dir=""):
    base_name = os.path.basename(file_path)
    file_dir = os.path.dirname(file_path)
    
    # make target dir
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    # cache resource
    if cache_resource:
        # download to cache dir
        file_cache_dir = f'{cache_dir}/{file_dir}'
        file_cache_path = f'{file_cache_dir}/{base_name}'
        if not os.path.exists(file_cache_dir):
            os.makedirs(file_cache_dir)
        if not os.path.exists(file_cache_path):
            _start_time = time.time()
            urllib.request.urlretrieve(url, file_cache_path)
            _end_time = time.time()
            print(f'[Download to cache] {url} -> {file_cache_path}')
            print("Elapsed time: {:.2f} seconds".format(_end_time - _start_time))
        else:
            print(f'[Download to cache] file exists: {file_cache_path}')
            
        # copy from cache
        file_path = f'{file_dir}/{base_name}'
        if not os.path.exists(file_path):
            shutil.copy(file_cache_path, file_path)
            print(f'[Copy from cache] {file_cache_path} -> {file_path}')
        else:
            print(f'[Copy from cache] file exists: {file_path}')
        
    # download directly
    else:
        file_path = f'{file_dir}/{base_name}'
        if not os.path.exists(file_path):
            print(f"[Download directly] start downloading ... \n   {url}")
            _start_time = time.time()
            urllib.request.urlretrieve(url, file_path)
            _end_time = time.time()
            print(f'[Download directly] {url} -> {file_path}')
            print("Elapsed time: {:.2f} seconds".format(_end_time - _start_time))
            
        else:
            print(f'[Download directly] file exists: {file_path}')



def download_models():
    
    model_root = "models"
    with open("config/model_list.json", "r", encoding="utf-8") as model_config_file:
        model_config = json.loads(model_config_file.read())
        
        all_type_models = model_config["models"]
        model_paths = model_config["model_paths"]
        
        # for each type of models download
        for model_type, model_type_info in all_type_models.items():
            
            model_dir = f"{model_root}/{model_paths.get(model_type, model_type)}"
            
            for model_name, model_info in model_type_info.items():
                url = model_info["url"]
                file_path = model_info["path"]
                file_path = f"{model_dir}/{file_path}"
                try:
                    ab_download_resource(url, file_path)
                except Exception as e:
                    print(f"[DownloadModels] FAIL. unexpected exception: {e}. \n{traceback.format_exc()}")

    return 



def aiyo_proj_init():
    # plugin config

    plugin_repos = []
    with open('config/default_plugins.json', 'r', encoding='utf-8') as config_file:
        default_plg_config = json.loads(config_file.read())
    plugin_repos = default_plg_config['plugins']

    # other config
    failed_retry = 3
    ext_path = 'custom_nodes'


    # clone plugin repos
    try_cnt = failed_retry
    while try_cnt > 0:
        try_cnt = try_cnt - 1
        
        try:
            # ==================== prepare extensions ============================

            for plg_info in plugin_repos:
                git_url = plg_info['files'][0]
                repo_name = git_url.split("/")[-1].split(".")[0]
                
                try:
                    local_path = f'{ext_path}/{repo_name}'
                    print(f'Cloning repo from: {git_url}')
                    print(f'    to: {local_path}')
                    if not os.path.exists(local_path):
                    
                        repo.Repo.clone_from(git_url, local_path, recursive=True)
                        
                        rep_cm = repo_name.get("commit", None)
                        if rep_cm is not None and rep_cm != "":
                            cur_rep = repo.Repo(local_path)
                            new_branch = cur_rep.create_head("aiyoh_work", rep_cm)
                            cur_rep.head.reference = new_branch
                            cur_rep.head.reset(index=True, working_tree=True)
                            
                    
                        req_txt = f"{local_path}/requirements.txt"
                        if os.path.exists(req_txt):
                            subprocess.check_call(['pip', 'install', '-r', req_txt])
                                
                        else:
                            print(f"Clone {git_url}:  repo exist.")
                        
                except Exception as e:  
                    print(f'Clone {git_url} FAIL.  network error.')
                    print(e.args)
                print(f'Clone repo DONE: {git_url}')

        except Exception as e:
            print(f"Exception: {e}")



    try_cnt = failed_retry
    while try_cnt > 0:
        try_cnt = try_cnt - 1
        
        download_models()



if __name__ == "__main__":
    aiyo_proj_init()
    
    