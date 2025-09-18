import json
import logging
import os
import shutil

import pytest

from comfy.cli_args import args

_gitattributes = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
"""


@pytest.mark.asyncio
async def test_known_repos(tmp_path_factory):
    prev_hub_cache = os.getenv("HF_HUB_CACHE")
    os.environ["HF_HUB_CACHE"] = str(tmp_path_factory.mktemp("huggingface_root_cache"))

    from comfy.cmd import folder_paths
    from comfy.component_model.folder_path_types import FolderPathsTuple
    from comfy.model_downloader import get_huggingface_repo_list, \
        get_or_download_huggingface_repo, _get_cache_hits, _delete_repo_from_huggingface_cache
    from comfy.model_downloader import KNOWN_HUGGINGFACE_MODEL_REPOS

    test_cache_dir = tmp_path_factory.mktemp("huggingface_cache")
    test_local_dir = tmp_path_factory.mktemp("huggingface_locals")
    test_repo_id = "doctorpangloss/comfyui_downloader_test"
    prev_huggingface = folder_paths.folder_names_and_paths["huggingface"]
    prev_huggingface_cache = folder_paths.folder_names_and_paths["huggingface_cache"]

    _delete_repo_from_huggingface_cache(test_repo_id)
    _delete_repo_from_huggingface_cache(test_repo_id, test_cache_dir)
    args.disable_known_models = False
    try:
        folder_paths.folder_names_and_paths["huggingface"] = FolderPathsTuple("huggingface", [test_local_dir], {""})
        folder_paths.folder_names_and_paths["huggingface_cache"] = FolderPathsTuple("huggingface_cache", [test_cache_dir], {""})

        cache_hits, locals_hits = _get_cache_hits([test_cache_dir], [test_local_dir], test_repo_id)
        assert len(cache_hits) == 0, "not downloaded yet"
        assert len(locals_hits) == 0, "not downloaded yet"

        # test downloading the repo and observing a cache hit on second access
        try:
            KNOWN_HUGGINGFACE_MODEL_REPOS.remove(test_repo_id)
            logging.error("unexpected, the test_repo_id was already in the KNOWN_HUGGINGFACE_MODEL_REPOS symbol")
        except KeyError:
            known_repos = get_huggingface_repo_list()
            assert test_repo_id not in known_repos

        # best to import this at the time that it is run, not when the test is initialized
        KNOWN_HUGGINGFACE_MODEL_REPOS.add(test_repo_id)
        known_repos = get_huggingface_repo_list()
        assert test_repo_id in known_repos

        cache_hits, locals_hits = _get_cache_hits([test_cache_dir], [test_local_dir], test_repo_id)
        assert len(cache_hits) == len(locals_hits) == 0, "not downloaded yet"

        # download to cache
        path = get_or_download_huggingface_repo(test_repo_id)
        assert path is not None

        cache_hits, locals_hits = _get_cache_hits([test_cache_dir], [test_local_dir], test_repo_id)
        assert len(cache_hits) == 1, "should have downloaded to cache"
        assert len(locals_hits) == 0, "should not have downloaded to a local dir"

        # load from cache
        args.disable_known_models = True
        path = get_or_download_huggingface_repo(test_repo_id)
        assert path is not None, "should have used local path"

        # test deleting from cache
        _delete_repo_from_huggingface_cache(test_repo_id)
        _delete_repo_from_huggingface_cache(test_repo_id, test_cache_dir)
        cache_hits, locals_hits = _get_cache_hits([test_cache_dir], [test_local_dir], test_repo_id)
        assert len(cache_hits) == 0, "should have deleted from the cache"
        assert len(locals_hits) == 0, "should not have downloaded to a local dir"

        # test fails to download
        path = get_or_download_huggingface_repo(test_repo_id)
        assert path is None, "should not have downloaded since disable_known_models is True"
        args.disable_known_models = False

        # download to local dir
        args.force_hf_local_dir_mode = True
        path = get_or_download_huggingface_repo(test_repo_id)
        assert path is not None
        cache_hits, locals_hits = _get_cache_hits([test_cache_dir], [test_local_dir], test_repo_id)
        assert len(cache_hits) == 0
        assert len(locals_hits) == 1, "should have downloaded to local dir"

        # test loads from local dir
        args.disable_known_models = True
        path = get_or_download_huggingface_repo(test_repo_id)
        assert path is not None

        # test deleting local dir
        expected_path = os.path.join(test_local_dir, test_repo_id)
        shutil.rmtree(expected_path)
        cache_hits, locals_hits = _get_cache_hits([test_cache_dir], [test_local_dir], test_repo_id)
        assert len(cache_hits) == 0
        assert len(locals_hits) == 0
        path = get_or_download_huggingface_repo(test_repo_id)
        assert path is None, "should not download repo into local dir"

        # recreating the test repo should be valid
        os.makedirs(expected_path)
        with open(os.path.join(expected_path, "test.txt"), "wt") as f:
            f.write("OK")
        with open(os.path.join(expected_path, ".gitattributes"), "wt") as f:
            f.write(_gitattributes)

        args.disable_known_models = False
        # expect local hit
        cache_hits, locals_hits = _get_cache_hits([test_cache_dir], [test_local_dir], test_repo_id)
        assert len(cache_hits) == 0
        assert len(locals_hits) == 1

        # should not download
        path = get_or_download_huggingface_repo(test_repo_id)
        assert path is not None
    finally:
        _delete_repo_from_huggingface_cache(test_repo_id)
        _delete_repo_from_huggingface_cache(test_repo_id, test_cache_dir)
        if test_repo_id in KNOWN_HUGGINGFACE_MODEL_REPOS:
            KNOWN_HUGGINGFACE_MODEL_REPOS.remove(test_repo_id)
        folder_paths.folder_names_and_paths["huggingface"] = prev_huggingface
        folder_paths.folder_names_and_paths["huggingface_cache"] = prev_huggingface_cache
        if prev_hub_cache is None and "HF_HUB_CACHE" in os.environ:
            os.environ.pop("HF_HUB_CACHE")
        elif prev_hub_cache is not None:
            os.environ["HF_HUB_CACHE"] = prev_hub_cache
        args.force_hf_local_dir_mode = False
        args.disable_known_models = False


@pytest.mark.asyncio
async def test_huggingface_alternate_filenames_in_combo():
    from comfy.model_downloader import get_filename_list_with_downloadable
    from comfy.model_downloader_types import HuggingFile

    # 2. Define a HuggingFile with alternate filenames
    main_filename = "model.safetensors"
    alternate_filename = "alt_model.safetensors"
    known_file = HuggingFile(
        repo_id="test/repo",
        filename=main_filename,
        alternate_filenames=(alternate_filename,)
    )

    # 3. Get the list of files as the UI would
    filename_list = get_filename_list_with_downloadable("__xxx___", known_files=[known_file])

    # 4. Assert that both the main and alternate filenames are present
    assert main_filename in filename_list
    assert alternate_filename not in filename_list, "Alternate filename should not be in the list returned by get_filename_list_with_downloadable"
    assert alternate_filename in filename_list.view_for_validation(), "Alternate filename should not be in the list returned by get_filename_list_with_downloadable"

    assert json.dumps(filename_list) == "[\"model.safetensors\"]"
