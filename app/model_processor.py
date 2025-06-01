import hashlib
import os
import logging
import time
from app.database.models import Model
from app.database.db import create_session
from folder_paths import get_relative_path


class ModelProcessor:
    def _validate_path(self, model_path):
        try:
            if not os.path.exists(model_path):
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

    def _hash_file(self, model_path):
        try:
            h = hashlib.sha256()
            with open(model_path, "rb", buffering=0) as f:
                b = bytearray(128 * 1024)
                mv = memoryview(b)
                while n := f.readinto(mv):
                    h.update(mv[:n])
            return h.hexdigest()
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

    def _update_database(
        self, session, model_type, model_relative_path, model_hash, model=None
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
                )
                session.add(model)

            model.hash = model_hash
            session.commit()
            return model
        except Exception as e:
            logging.error(
                f"Error updating database for {model_relative_path}: {str(e)}"
            )

    def process_file(self, model_path):
        try:
            result = self._validate_path(model_path)
            if not result:
                return
            model_type, model_relative_path = result

            with create_session() as session:
                existing_model = self._get_existing_model(
                    session, model_type, model_relative_path
                )
                if existing_model and existing_model.hash:
                    # File exists with hash, no need to process
                    return existing_model

                start_time = time.time()
                logging.info(f"Hashing model {model_relative_path}")
                model_hash = self._hash_file(model_path)
                if not model_hash:
                    return
                logging.info(
                    f"Model hash: {model_hash} (duration: {time.time() - start_time} seconds)"
                )

                return self._update_database(session, model_type, model_relative_path, model_hash)
        except Exception as e:
            logging.error(f"Error processing model file {model_path}: {str(e)}")

    def retrieve_hash(self, model_path, model_type=None):
        try:
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


model_processor = ModelProcessor()
