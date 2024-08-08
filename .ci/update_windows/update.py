import pygit2
from datetime import datetime
import sys
import os
import shutil
import filecmp

def pull(repo, remote_name='origin', branch='master'):
    for remote in repo.remotes:
        if remote.name == remote_name:
            remote.fetch()
            remote_master_id = repo.lookup_reference('refs/remotes/origin/%s' % (branch)).target
            merge_result, _ = repo.merge_analysis(remote_master_id)
            # Up to date, do nothing
            if merge_result & pygit2.GIT_MERGE_ANALYSIS_UP_TO_DATE:
                return
            # We can just fastforward
            elif merge_result & pygit2.GIT_MERGE_ANALYSIS_FASTFORWARD:
                repo.checkout_tree(repo.get(remote_master_id))
                try:
                    master_ref = repo.lookup_reference('refs/heads/%s' % (branch))
                    master_ref.set_target(remote_master_id)
                except KeyError:
                    repo.create_branch(branch, repo.get(remote_master_id))
                repo.head.set_target(remote_master_id)
            elif merge_result & pygit2.GIT_MERGE_ANALYSIS_NORMAL:
                repo.merge(remote_master_id)

                if repo.index.conflicts is not None:
                    for conflict in repo.index.conflicts:
                        print('Conflicts found in:', conflict[0].path)
                    raise AssertionError('Conflicts, ahhhhh!!')

                user = repo.default_signature
                tree = repo.index.write_tree()
                commit = repo.create_commit('HEAD',
                                            user,
                                            user,
                                            'Merge!',
                                            tree,
                                            [repo.head.target, remote_master_id])
                # We need to do this or git CLI will think we are still merging.
                repo.state_cleanup()
            else:
                raise AssertionError('Unknown merge analysis result')

pygit2.option(pygit2.GIT_OPT_SET_OWNER_VALIDATION, 0)
repo_path = str(sys.argv[1])
repo = pygit2.Repository(repo_path)
ident = pygit2.Signature('comfyui', 'comfy@ui')
try:
    print("stashing current changes")
    repo.stash(ident)
except KeyError:
    print("nothing to stash")
backup_branch_name = 'backup_branch_{}'.format(datetime.today().strftime('%Y-%m-%d_%H_%M_%S'))
print("creating backup branch: {}".format(backup_branch_name))
try:
    repo.branches.local.create(backup_branch_name, repo.head.peel())
except:
    pass

print("checking out master branch")
branch = repo.lookup_branch('master')
if branch is None:
    ref = repo.lookup_reference('refs/remotes/origin/master')
    repo.checkout(ref)
    branch = repo.lookup_branch('master')
    if branch is None:
        repo.create_branch('master', repo.get(ref.target))
else:
    ref = repo.lookup_reference(branch.name)
    repo.checkout(ref)

print("pulling latest changes")
pull(repo)

print("Done!")

self_update = True
if len(sys.argv) > 2:
    self_update = '--skip_self_update' not in sys.argv

update_py_path = os.path.realpath(__file__)
repo_update_py_path = os.path.join(repo_path, ".ci/update_windows/update.py")

cur_path = os.path.dirname(update_py_path)


req_path = os.path.join(cur_path, "current_requirements.txt")
repo_req_path = os.path.join(repo_path, "requirements.txt")


def files_equal(file1, file2):
    try:
        return filecmp.cmp(file1, file2, shallow=False)
    except:
        return False

def file_size(f):
    try:
        return os.path.getsize(f)
    except:
        return 0


if self_update and not files_equal(update_py_path, repo_update_py_path) and file_size(repo_update_py_path) > 10:
    shutil.copy(repo_update_py_path, os.path.join(cur_path, "update_new.py"))
    exit()

if not os.path.exists(req_path) or not files_equal(repo_req_path, req_path):
    import subprocess
    try:
        subprocess.check_call([sys.executable, '-s', '-m', 'pip', 'install', '-r', repo_req_path])
        shutil.copy(repo_req_path, req_path)
    except:
        pass
