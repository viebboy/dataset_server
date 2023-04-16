"""
__init__.py: init file
----------------------


* Copyright: 2022 Dat Tran
* Authors: Dat Tran (viebboy@gmail.com)
* Date: 2023-04-06
* Version: 0.0.1

This is part of the dataset_server (github.com/viebboy/dataset_server)

License
-------
Apache License 2.0


"""

# check for update
import time
try:
    from loguru import logger
    import git
    import pkg_resources
    import requests
    import os
    has_dep = True
except Exception as error:
    print('package dataset_server WARNING: (re)install the dependencies in requirements.txt')
    has_dep = False

# root dir is for user configuration
STATUS_FILE = os.path.join(os.path.expanduser('~'), '.dataset_server.status')
has_status_file = os.path.exists(STATUS_FILE)

if has_status_file:
    last_modified = os.path.getmtime(STATUS_FILE)
    days_since_modified = (time.time() - last_modified) / (24 * 3600)
    if days_since_last_modified > 1:
        require_checking = True
    else:
        require_checking = False
else:
    require_checking = True


if has_dep and require_checking:
    with open(STATUS_FILE, 'w') as fid:
        fid.write('')

    package = 'dataset_server'
    branch = 'multiprocessing'

    remote_repo = f"https://github.com/viebboy/{package}"

    # Get the local path
    try:
        src_path = pkg_resources.resource_filename(package, "")
        is_installed = True
    except Exception:
        is_installed = False

    if is_installed:
        local_path = os.path.dirname(src_path)
        local_repo = git.Repo(local_path)

        # Get the current commit hash of the local repository
        local_commit = local_repo.head.object.hexsha

        # Query the GitHub API for the latest commit on the main branch of the remote repository
        api_url = f"https://api.github.com/repos/{remote_repo.split('/')[-2]}/{remote_repo.split('/')[-1]}"
        response = requests.get(f"{api_url}/branches/{branch}")
        response.raise_for_status()
        remote_commit = response.json()["commit"]["sha"]

        # Compare the local and remote commit hashes
        if local_commit != remote_commit:
            logger.warning(
                f'local commit diverges from remote commit in package {package}. Please consider updating'
            )
            logger.warning(f'local commit on branch {branch}: {local_commit}')
            logger.warning(f'remote commit on branch {branch}: {remote_commit}')


from dataset_server.apis import DataLoader
__all__ = ['DataLoader']
