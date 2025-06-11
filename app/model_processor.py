import os
import logging
import time

import requests
from tqdm import tqdm
from folder_paths import get_relative_path, get_full_path
from app.database.db import create_session, dependencies_available, can_create_session
import blake3
import comfy.utils


if dependencies_available():
    from app.database.models import Model


class ModelProcessor:
    def _validate_path(self, model_path):
        try:
            if not self._file_exists(model_path):
                logging.error(f"Model file not found: {model_path}")
                return None

            result = get_relative_path(model_path)
            if not result:
                logging.error(
                    f"Model file not in a recognized model directory: {model_path}"
                )
                return None

            return result
        except Exception as e:
            logging.error(f"Error validating model path {model_path}: {str(e)}")
            return None

    def _file_exists(self, path):
        """Check if a file exists."""
        return os.path.exists(path)

    def _get_file_size(self, path):
        """Get file size."""
        return os.path.getsize(path)

    def _get_hasher(self):
        return blake3.blake3()

    def _hash_file(self, model_path):
        try:
            hasher = self._get_hasher()
            with open(model_path, "rb", buffering=0) as f:
                b = bytearray(128 * 1024)
                mv = memoryview(b)
                while n := f.readinto(mv):
                    hasher.update(mv[:n])
            return hasher.hexdigest()
        except Exception as e:
            logging.error(f"Error hashing file {model_path}: {str(e)}")
            return None

    def _get_existing_model(self, session, model_type, model_relative_path):
        return (
            session.query(Model)
            .filter(Model.type == model_type)
            .filter(Model.path == model_relative_path)
            .first()
        )

    def _ensure_source_url(self, session, model, source_url):
        if model.source_url is None:
            model.source_url = source_url
            session.commit()

    def _update_database(
        self,
        session,
        model_type,
        model_path,
        model_relative_path,
        model_hash,
        model,
        source_url,
    ):
        try:
            if not model:
                model = self._get_existing_model(
                    session, model_type, model_relative_path
                )

            if not model:
                model = Model(
                    path=model_relative_path,
                    type=model_type,
                    file_name=os.path.basename(model_path),
                )
                session.add(model)

            model.file_size = self._get_file_size(model_path)
            model.hash = model_hash
            if model_hash:
                model.hash_algorithm = "blake3"
            model.source_url = source_url

            session.commit()
            return model
        except Exception as e:
            logging.error(
                f"Error updating database for {model_relative_path}: {str(e)}"
            )

    def process_file(self, model_path, source_url=None, model_hash=None):
        """
        Process a model file and update the database with metadata.
        If the file already exists and matches the database, it will not be processed again.
        Returns the model object or if an error occurs, returns None.
        """
        try:
            if not can_create_session():
                return

            result = self._validate_path(model_path)
            if not result:
                return
            model_type, model_relative_path = result

            with create_session() as session:
                session.expire_on_commit = False

                existing_model = self._get_existing_model(
                    session, model_type, model_relative_path
                )
                if (
                    existing_model
                    and existing_model.hash
                    and existing_model.file_size == self._get_file_size(model_path)
                ):
                    # File exists with hash and same size, no need to process
                    self._ensure_source_url(session, existing_model, source_url)
                    return existing_model

                if model_hash:
                    model_hash = model_hash.lower()
                    logging.info(f"Using provided hash: {model_hash}")
                else:
                    start_time = time.time()
                    logging.info(f"Hashing model {model_relative_path}")
                    model_hash = self._hash_file(model_path)
                    if not model_hash:
                        return
                    logging.info(
                        f"Model hash: {model_hash} (duration: {time.time() - start_time} seconds)"
                    )

                return self._update_database(
                    session,
                    model_type,
                    model_path,
                    model_relative_path,
                    model_hash,
                    existing_model,
                    source_url,
                )
        except Exception as e:
            logging.error(f"Error processing model file {model_path}: {str(e)}")
            return None

    def retrieve_model_by_hash(self, model_hash, model_type=None, session=None):
        """
        Retrieve a model file from the database by hash and optionally by model type.
        Returns the model object or None if the model doesnt exist or an error occurs.
        """
        try:
            if not can_create_session():
                return

            dispose_session = False

            if session is None:
                session = create_session()
                dispose_session = True

            model = session.query(Model).filter(Model.hash == model_hash)
            if model_type is not None:
                model = model.filter(Model.type == model_type)
            return model.first()
        except Exception as e:
            logging.error(f"Error retrieving model by hash {model_hash}: {str(e)}")
            return None
        finally:
            if dispose_session:
                session.close()

    def retrieve_hash(self, model_path, model_type=None):
        """
        Retrieve the hash of a model file from the database.
        Returns the hash or None if the model doesnt exist or an error occurs.
        """
        try:
            if not can_create_session():
                return

            if model_type is not None:
                result = self._validate_path(model_path)
                if not result:
                    return None
            model_type, model_relative_path = result

            with create_session() as session:
                model = self._get_existing_model(
                    session, model_type, model_relative_path
                )
                if model and model.hash:
                    return model.hash
                return None
        except Exception as e:
            logging.error(f"Error retrieving hash for {model_path}: {str(e)}")
            return None

    def _validate_file_extension(self, file_name):
        """Validate that the file extension is supported."""
        extension = os.path.splitext(file_name)[1]
        if extension not in (".safetensors", ".sft", ".txt", ".csv", ".json", ".yaml"):
            raise ValueError(f"Unsupported unsafe file for download: {file_name}")

    def _check_existing_file(self, model_type, file_name, expected_hash):
        """Check if file exists and has correct hash."""
        destination_path = get_full_path(model_type, file_name, allow_missing=True)
        if self._file_exists(destination_path):
            model = self.process_file(destination_path)
            if model and (expected_hash is None or model.hash == expected_hash):
                logging.debug(
                    f"File {destination_path} already exists in the database and has the correct hash or no hash was provided."
                )
                return destination_path
            else:
                raise ValueError(
                    f"File {destination_path} exists with hash {model.hash if model else 'unknown'} but expected {expected_hash}. Please delete the file and try again."
                )
        return None

    def _check_existing_file_by_hash(self, hash, type, url):
        """Check if a file with the given hash exists in the database and on disk."""
        hash = hash.lower()
        with create_session() as session:
            model = self.retrieve_model_by_hash(hash, type, session)
            if model:
                existing_path = get_full_path(type, model.path)
                if existing_path:
                    logging.debug(
                        f"File {model.path} already exists in the database at {existing_path}"
                    )
                    self._ensure_source_url(session, model, url)
                    return existing_path
                else:
                    logging.debug(
                        f"File {model.path} exists in the database but not on disk"
                    )
        return None

    def _download_file(self, url, destination_path, hasher):
        """Download a file and update the hasher with its contents."""
        response = requests.get(url, stream=True)
        logging.info(f"Downloading {url} to {destination_path}")

        with open(destination_path, "wb") as f:
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                pbar = comfy.utils.ProgressBar(total_size)
            else:
                pbar = None
            with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                for chunk in response.iter_content(chunk_size=128 * 1024):
                    if chunk:
                        f.write(chunk)
                        hasher.update(chunk)
                        progress_bar.update(len(chunk))
                        if pbar:
                            pbar.update(len(chunk))

    def _verify_downloaded_hash(self, calculated_hash, expected_hash, destination_path):
        """Verify that the downloaded file has the expected hash."""
        if expected_hash is not None and calculated_hash != expected_hash:
            self._remove_file(destination_path)
            raise ValueError(
                f"Downloaded file hash {calculated_hash} does not match expected hash {expected_hash}"
            )

    def _remove_file(self, file_path):
        """Remove a file from disk."""
        os.remove(file_path)

    def ensure_downloaded(self, type, url, desired_file_name, hash=None):
        """
        Ensure a model file is downloaded and has the correct hash.
        Returns the path to the downloaded file.
        """
        logging.debug(
            f"Ensuring {type} file is downloaded. URL='{url}' Destination='{desired_file_name}' Hash='{hash}'"
        )

        # Validate file extension
        self._validate_file_extension(desired_file_name)

        # Check if file exists with correct hash
        if hash:
            existing_path = self._check_existing_file_by_hash(hash, type, url)
            if existing_path:
                return existing_path

        # Check if file exists locally
        destination_path = get_full_path(type, desired_file_name, allow_missing=True)
        existing_path = self._check_existing_file(type, desired_file_name, hash)
        if existing_path:
            return existing_path

        # Download the file
        hasher = self._get_hasher()
        self._download_file(url, destination_path, hasher)

        # Verify hash
        calculated_hash = hasher.hexdigest()
        self._verify_downloaded_hash(calculated_hash, hash, destination_path)

        # Update database
        self.process_file(destination_path, url, calculated_hash)

        # TODO: Notify frontend to reload models

        return destination_path


model_processor = ModelProcessor()
