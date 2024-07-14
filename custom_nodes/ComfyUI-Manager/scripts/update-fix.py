import git

commit_hash = "a361cc1"

repo = git.Repo('.')

if repo.is_dirty():
    repo.git.stash()

repo.git.update_ref("refs/remotes/origin/main", commit_hash)
repo.remotes.origin.fetch()
repo.git.pull("origin", "main")
