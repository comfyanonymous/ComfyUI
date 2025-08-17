import os
import aiohttp
import logging
import toml
import git
import zipfile
import shutil
import tempfile
from aiohttp import web
from typing import Dict, List


logger = logging.getLogger(__name__)

class UpdateRoutes:
    """Routes for handling plugin update checks"""
    
    @staticmethod
    def setup_routes(app):
        """Register update check routes"""
        app.router.add_get('/api/check-updates', UpdateRoutes.check_updates)
        app.router.add_get('/api/version-info', UpdateRoutes.get_version_info)
        app.router.add_post('/api/perform-update', UpdateRoutes.perform_update)
    
    @staticmethod
    async def check_updates(request):
        """
        Check for plugin updates by comparing local version with GitHub
        Returns update status and version information
        """
        try:
            nightly = request.query.get('nightly', 'false').lower() == 'true'
            
            # Read local version from pyproject.toml
            local_version = UpdateRoutes._get_local_version()
            
            # Get git info (commit hash, branch)
            git_info = UpdateRoutes._get_git_info()

            # Fetch remote version from GitHub
            if nightly:
                remote_version, changelog = await UpdateRoutes._get_nightly_version()
            else:
                remote_version, changelog = await UpdateRoutes._get_remote_version()
            
            # Compare versions
            if nightly:
                # For nightly, compare commit hashes
                update_available = UpdateRoutes._compare_nightly_versions(git_info, remote_version)
            else:
                # For stable, compare semantic versions
                update_available = UpdateRoutes._compare_versions(
                    local_version.replace('v', ''), 
                    remote_version.replace('v', '')
                )
            
            return web.json_response({
                'success': True,
                'current_version': local_version,
                'latest_version': remote_version,
                'update_available': update_available,
                'changelog': changelog,
                'git_info': git_info,
                'nightly': nightly
            })
            
        except Exception as e:
            logger.error(f"Failed to check for updates: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            })
        
    @staticmethod
    async def get_version_info(request):
        """
        Returns the current version in the format 'version-short_hash'
        """
        try:
            # Read local version from pyproject.toml
            local_version = UpdateRoutes._get_local_version().replace('v', '')
            
            # Get git info (commit hash, branch)
            git_info = UpdateRoutes._get_git_info()
            short_hash = git_info['short_hash']
            
            # Format: version-short_hash
            version_string = f"{local_version}-{short_hash}"
            
            return web.json_response({
                'success': True,
                'version': version_string
            })
            
        except Exception as e:
            logger.error(f"Failed to get version info: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            })
    
    @staticmethod
    async def perform_update(request):
        """
        Perform Git-based update to latest release tag or main branch.
        If .git is missing, fallback to ZIP download.
        """
        try:
            body = await request.json() if request.has_body else {}
            nightly = body.get('nightly', False)

            current_dir = os.path.dirname(os.path.abspath(__file__))
            plugin_root = os.path.dirname(os.path.dirname(current_dir))

            settings_path = os.path.join(plugin_root, 'settings.json')
            settings_backup = None
            if os.path.exists(settings_path):
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings_backup = f.read()
                logger.info("Backed up settings.json")

            git_folder = os.path.join(plugin_root, '.git')
            if os.path.exists(git_folder):
                # Git update
                success, new_version = await UpdateRoutes._perform_git_update(plugin_root, nightly)
            else:
                # Fallback: Download ZIP and replace files
                success, new_version = await UpdateRoutes._download_and_replace_zip(plugin_root)

            if settings_backup and success:
                with open(settings_path, 'w', encoding='utf-8') as f:
                    f.write(settings_backup)
                logger.info("Restored settings.json")

            if success:
                return web.json_response({
                    'success': True,
                    'message': f'Successfully updated to {new_version}',
                    'new_version': new_version
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to complete update'
                })

        except Exception as e:
            logger.error(f"Failed to perform update: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            })

    @staticmethod
    async def _download_and_replace_zip(plugin_root: str) -> tuple[bool, str]:
        """
        Download latest release ZIP from GitHub and replace plugin files.
        Skips settings.json. Writes extracted file list to .tracking.
        """
        repo_owner = "willmiao"
        repo_name = "ComfyUI-Lora-Manager"
        github_api = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(github_api) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to fetch release info: {resp.status}")
                        return False, ""
                    data = await resp.json()
                    zip_url = data.get("zipball_url")
                    version = data.get("tag_name", "unknown")

                # Download ZIP
                async with session.get(zip_url) as zip_resp:
                    if zip_resp.status != 200:
                        logger.error(f"Failed to download ZIP: {zip_resp.status}")
                        return False, ""
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                        tmp_zip.write(await zip_resp.read())
                        zip_path = tmp_zip.name

                UpdateRoutes._clean_plugin_folder(plugin_root, skip_files=['settings.json'])

                # Extract ZIP to temp dir
                with tempfile.TemporaryDirectory() as tmp_dir:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(tmp_dir)
                    # Find extracted folder (GitHub ZIP contains a root folder)
                    extracted_root = next(os.scandir(tmp_dir)).path

                    # Copy files, skipping settings.json
                    for item in os.listdir(extracted_root):
                        src = os.path.join(extracted_root, item)
                        dst = os.path.join(plugin_root, item)
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst, ignore=shutil.ignore_patterns('settings.json'))
                        else:
                            if item == 'settings.json':
                                continue
                            shutil.copy2(src, dst)

                    # Write .tracking file: list all files under extracted_root, relative to extracted_root
                    # for ComfyUI Manager to work properly
                    tracking_info_file = os.path.join(plugin_root, '.tracking')
                    tracking_files = []
                    for root, dirs, files in os.walk(extracted_root):
                        for file in files:
                            rel_path = os.path.relpath(os.path.join(root, file), extracted_root)
                            tracking_files.append(rel_path.replace("\\", "/"))
                    with open(tracking_info_file, "w", encoding='utf-8') as file:
                        file.write('\n'.join(tracking_files))

                os.remove(zip_path)
                logger.info(f"Updated plugin via ZIP to {version}")
                return True, version

        except Exception as e:
            logger.error(f"ZIP update failed: {e}", exc_info=True)
            return False, ""
        
    def _clean_plugin_folder(plugin_root, skip_files=None):
        skip_files = skip_files or []
        for item in os.listdir(plugin_root):
            if item in skip_files:
                continue
            path = os.path.join(plugin_root, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    
    @staticmethod
    async def _get_nightly_version() -> tuple[str, List[str]]:
        """
        Fetch latest commit from main branch
        """
        repo_owner = "willmiao"
        repo_name = "ComfyUI-Lora-Manager"
        
        # Use GitHub API to fetch the latest commit from main branch
        github_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/commits/main"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(github_url, headers={'Accept': 'application/vnd.github+json'}) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch GitHub commit: {response.status}")
                        return "main", []
                    
                    data = await response.json()
                    commit_sha = data.get('sha', '')[:7]  # Short hash
                    commit_message = data.get('commit', {}).get('message', '')
                    
                    # Format as "main-{short_hash}"
                    version = f"main-{commit_sha}"
                    
                    # Use commit message as changelog
                    changelog = [commit_message] if commit_message else []
                    
                    return version, changelog
        
        except Exception as e:
            logger.error(f"Error fetching nightly version: {e}", exc_info=True)
            return "main", []
    
    @staticmethod
    def _compare_nightly_versions(local_git_info: Dict[str, str], remote_version: str) -> bool:
        """
        Compare local commit hash with remote main branch
        """
        try:
            local_hash = local_git_info.get('short_hash', 'unknown')
            if local_hash == 'unknown':
                return True  # Assume update available if we can't get local hash
            
            # Extract remote hash from version string (format: "main-{hash}")
            if '-' in remote_version:
                remote_hash = remote_version.split('-')[-1]
                return local_hash != remote_hash
            
            return True  # Default to update available
            
        except Exception as e:
            logger.error(f"Error comparing nightly versions: {e}")
            return False
    
    @staticmethod
    async def _perform_git_update(plugin_root: str, nightly: bool = False) -> tuple[bool, str]:
        """
        Perform Git-based update using GitPython
        
        Args:
            plugin_root: Path to the plugin root directory
            nightly: Whether to update to main branch or latest release
            
        Returns:
            tuple: (success, new_version)
        """
        try:
            # Open the Git repository
            repo = git.Repo(plugin_root)
            
            # Fetch latest changes
            origin = repo.remotes.origin
            origin.fetch()
            
            if nightly:
                # Switch to main branch and pull latest
                main_branch = 'main'
                if main_branch not in [branch.name for branch in repo.branches]:
                    # Create local main branch if it doesn't exist
                    repo.create_head(main_branch, origin.refs.main)
                
                repo.heads[main_branch].checkout()
                origin.pull(main_branch)
                
                # Get new commit hash
                new_version = f"main-{repo.head.commit.hexsha[:7]}"
                
            else:
                # Get latest release tag
                tags = sorted(repo.tags, key=lambda t: t.commit.committed_datetime, reverse=True)
                if not tags:
                    logger.error("No tags found in repository")
                    return False, ""
                
                latest_tag = tags[0]
                
                # Checkout to latest tag
                repo.git.checkout(latest_tag.name)
                
                new_version = latest_tag.name
            
            logger.info(f"Successfully updated to {new_version}")
            return True, new_version
            
        except git.exc.GitError as e:
            logger.error(f"Git error during update: {e}")
            return False, ""
        except Exception as e:
            logger.error(f"Error during Git update: {e}")
            return False, ""
    
    @staticmethod
    def _get_local_version() -> str:
        """Get local plugin version from pyproject.toml"""
        try:
            # Find the plugin's pyproject.toml file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            plugin_root = os.path.dirname(os.path.dirname(current_dir))
            pyproject_path = os.path.join(plugin_root, 'pyproject.toml')
            
            # Read and parse the toml file
            if os.path.exists(pyproject_path):
                with open(pyproject_path, 'r', encoding='utf-8') as f:
                    project_data = toml.load(f)
                    version = project_data.get('project', {}).get('version', '0.0.0')
                    return f"v{version}"
            else:
                logger.warning(f"pyproject.toml not found at {pyproject_path}")
                return "v0.0.0"
        
        except Exception as e:
            logger.error(f"Failed to get local version: {e}", exc_info=True)
            return "v0.0.0"
    
    @staticmethod
    def _get_git_info() -> Dict[str, str]:
        """Get Git repository information"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        plugin_root = os.path.dirname(os.path.dirname(current_dir))

        git_info = {
            'commit_hash': 'unknown',
            'short_hash': 'stable',
            'branch': 'unknown',
            'commit_date': 'unknown'
        }

        try:
            # Check if we're in a git repository
            if not os.path.exists(os.path.join(plugin_root, '.git')):
                return git_info

            repo = git.Repo(plugin_root)
            commit = repo.head.commit
            git_info['commit_hash'] = commit.hexsha
            git_info['short_hash'] = commit.hexsha[:7]
            git_info['branch'] = repo.active_branch.name if not repo.head.is_detached else 'detached'
            git_info['commit_date'] = commit.committed_datetime.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Error getting git info: {e}")

        return git_info
    
    @staticmethod
    async def _get_remote_version() -> tuple[str, List[str]]:
        """
        Fetch remote version from GitHub
        Returns:
            tuple: (version string, changelog list)
        """
        repo_owner = "willmiao"
        repo_name = "ComfyUI-Lora-Manager"
        
        # Use GitHub API to fetch the latest release
        github_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases/latest"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(github_url, headers={'Accept': 'application/vnd.github+json'}) as response:
                    if response.status != 200:
                        logger.warning(f"Failed to fetch GitHub release: {response.status}")
                        return "v0.0.0", []
                    
                    data = await response.json()
                    version = data.get('tag_name', '')
                    if not version.startswith('v'):
                        version = f"v{version}"
                    
                    # Extract changelog from release notes
                    body = data.get('body', '')
                    changelog = UpdateRoutes._parse_changelog(body)
                    
                    return version, changelog
        
        except Exception as e:
            logger.error(f"Error fetching remote version: {e}", exc_info=True)
            return "v0.0.0", []
    
    @staticmethod
    def _parse_changelog(release_notes: str) -> List[str]:
        """
        Parse GitHub release notes to extract changelog items
        
        Args:
            release_notes: GitHub release notes markdown text
            
        Returns:
            List of changelog items
        """
        changelog = []
        
        # Simple parsing - extract bullet points
        lines = release_notes.split('\n')
        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered items
            if line.startswith('- ') or line.startswith('* '):
                item = line[2:].strip()
                if item:
                    changelog.append(item)
            # Match numbered items like "1. Item"
            elif len(line) > 2 and line[0].isdigit() and line[1:].startswith('. '):
                item = line[line.index('. ')+2:].strip()
                if item:
                    changelog.append(item)
        
        # If we couldn't parse specific items, use the whole text (limited)
        if not changelog and release_notes:
            # Limit to first 500 chars and add ellipsis
            summary = release_notes.strip()[:500]
            if len(release_notes) > 500:
                summary += "..."
            changelog.append(summary)
            
        return changelog
    
    @staticmethod
    def _compare_versions(version1: str, version2: str) -> bool:
        """
        Compare two semantic version strings
        Returns True if version2 is newer than version1
        Ignores any suffixes after '-' (e.g., -bugfix, -alpha)
        """
        try:
            # Clean version strings - remove any suffix after '-'
            v1_clean = version1.split('-')[0]
            v2_clean = version2.split('-')[0]
            
            # Split versions into components
            v1_parts = [int(x) for x in v1_clean.split('.')]
            v2_parts = [int(x) for x in v2_clean.split('.')]
            
            # Ensure both have 3 components (major.minor.patch)
            while len(v1_parts) < 3:
                v1_parts.append(0)
            while len(v2_parts) < 3:
                v2_parts.append(0)
            
            # Compare version components
            for i in range(3):
                if v2_parts[i] > v1_parts[i]:
                    return True
                elif v2_parts[i] < v1_parts[i]:
                    return False
            
            # Versions are equal
            return False
        except Exception as e:
            logger.error(f"Error comparing versions: {e}", exc_info=True)
            return False
