
import json, os

from git import repo
import subprocess


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



