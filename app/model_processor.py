import base64
from datetime import datetime
import glob
import hashlib
from io import BytesIO
import json
import logging
import os
import threading
import time
import comfy.utils
from app.database.models import Model
from app.database.db import create_session
from comfy.cli_args import args
from folder_paths import (
    filter_files_content_types,
    get_full_path,
    folder_names_and_paths,
    get_filename_list,
)
from PIL import Image
from urllib import request


def get_model_previews(
    filepath: str, check_metadata: bool = True
) -> list[str | BytesIO]:
    dirname = os.path.dirname(filepath)

    if not os.path.exists(dirname):
        return []

    basename = os.path.splitext(filepath)[0]
    match_files = glob.glob(f"{basename}.*", recursive=False)
    image_files = filter_files_content_types(match_files, "image")

    result: list[str | BytesIO] = []

    for filename in image_files:
        _basename = os.path.splitext(filename)[0]
        if _basename == basename:
            result.append(filename)
        if _basename == f"{basename}.preview":
            result.append(filename)

    if not check_metadata:
        return result

    safetensors_file = next(
        filter(lambda x: x.endswith(".safetensors"), match_files), None
    )
    safetensors_metadata = {}

    if safetensors_file:
        safetensors_filepath = os.path.join(dirname, safetensors_file)
        header = comfy.utils.safetensors_header(
            safetensors_filepath, max_size=8 * 1024 * 1024
        )
        if header:
            safetensors_metadata = json.loads(header)
    safetensors_images = safetensors_metadata.get("__metadata__", {}).get(
        "ssmd_cover_images", None
    )
    if safetensors_images:
        safetensors_images = json.loads(safetensors_images)
        for image in safetensors_images:
            result.append(BytesIO(base64.b64decode(image)))

    return result


class ModelProcessor:
    def __init__(self):
        self._thread = None
        self._lock = threading.Lock()
        self._run = False
        self.missing_models = []

    def run(self):
        if args.disable_model_processing:
            return

        if self._thread is None:
            # Lock to prevent multiple threads from starting
            with self._lock:
                self._run = True
                if self._thread is None:
                    self._thread = threading.Thread(target=self._process_models)
                    self._thread.daemon = True
                    self._thread.start()

    def populate_models(self, session):
        # Ensure database state matches filesystem

        existing_models = session.query(Model).all()

        for folder_name in folder_names_and_paths.keys():
            if folder_name == "custom_nodes" or folder_name == "configs":
                continue
            seen = set()
            files = get_filename_list(folder_name)

            for file in files:
                if file in seen:
                    logging.warning(f"Skipping duplicate named model: {file}")
                    continue
                seen.add(file)

                existing_model = None
                for model in existing_models:
                    if model.path == file and model.type == folder_name:
                        existing_model = model
                        break

                if existing_model:
                    # Model already exists in db, remove from list and skip
                    existing_models.remove(existing_model)
                    continue

                file_path = get_full_path(folder_name, file)

                model = Model(
                    path=file,
                    type=folder_name,
                    date_added=datetime.fromtimestamp(os.path.getctime(file_path)),
                )
                session.add(model)

        for model in existing_models:
            if not get_full_path(model.type, model.path):
                logging.warning(f"Model {model.path} not found")
                self.missing_models.append({"type": model.type, "path": model.path})

        session.commit()

    def _get_models(self, session):
        models = session.query(Model).filter(Model.hash == None).all()
        return models

    def _process_file(self, model_path):
        is_safetensors = model_path.endswith(".safetensors")
        metadata = {}
        h = hashlib.sha256()

        with open(model_path, "rb", buffering=0) as f:
            if is_safetensors:
                # Read header length (8 bytes)
                header_size_bytes = f.read(8)
                header_len = int.from_bytes(header_size_bytes, "little")
                h.update(header_size_bytes)

                # Read header
                header_bytes = f.read(header_len)
                h.update(header_bytes)
                try:
                    metadata = json.loads(header_bytes)
                except json.JSONDecodeError:
                    pass

            # Read rest of file
            b = bytearray(128 * 1024)
            mv = memoryview(b)
            while n := f.readinto(mv):
                h.update(mv[:n])

        return h.hexdigest(), metadata

    def _populate_info(self, model, metadata):
        model.title = metadata.get("modelspec.title", None)
        model.description = metadata.get("modelspec.description", None)
        model.architecture = metadata.get("modelspec.architecture", None)

    def _extract_image(self, model_path, metadata):
        # check if image already exists
        if len(get_model_previews(model_path, check_metadata=False)) > 0:
            return

        image_path = os.path.splitext(model_path)[0] + ".webp"
        if os.path.exists(image_path):
            return

        cover_images = metadata.get("ssmd_cover_images", None)
        image = None
        if cover_images:
            try:
                cover_images = json.loads(cover_images)
                if len(cover_images) > 0:
                    image_data = cover_images[0]
                    image = Image.open(BytesIO(base64.b64decode(image_data)))
            except Exception as e:
                logging.warning(
                    f"Error extracting cover image for model {model_path}: {e}"
                )

        if not image:
            thumbnail = metadata.get("modelspec.thumbnail", None)
            if thumbnail:
                try:
                    response = request.urlopen(thumbnail)
                    image = Image.open(response)
                except Exception as e:
                    logging.warning(
                        f"Error extracting thumbnail for model {model_path}: {e}"
                    )

        if image:
            image.thumbnail((512, 512))
            image.save(image_path)
            image.close()

    def _process_models(self):
        with create_session() as session:
            checked = set()
            self.populate_models(session)

            while self._run:
                self._run = False

                models = self._get_models(session)

                if len(models) == 0:
                    break

                for model in models:
                    # prevent looping on the same model if it crashes
                    if model.path in checked:
                        continue

                    checked.add(model.path)

                    try:
                        time.sleep(0)
                        now = time.time()
                        model_path = get_full_path(model.type, model.path)

                        if not model_path:
                            logging.warning(f"Model {model.path} not found")
                            self.missing_models.append(model.path)
                            continue

                        logging.debug(f"Processing model {model_path}")
                        hash, header = self._process_file(model_path)
                        logging.debug(
                            f"Processed model {model_path} in {time.time() - now} seconds"
                        )
                        model.hash = hash

                        if header:
                            metadata = header.get("__metadata__", None)

                            if metadata:
                                self._populate_info(model, metadata)
                                self._extract_image(model_path, metadata)

                        session.commit()
                    except Exception as e:
                        logging.error(f"Error processing model {model.path}: {e}")

        with self._lock:
            self._thread = None


model_processor = ModelProcessor()
