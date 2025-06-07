import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.model_processor import ModelProcessor
from app.database.models import Model, Base
import os

# Test data constants
TEST_MODEL_TYPE = "checkpoints"
TEST_URL = "http://example.com/model.safetensors"
TEST_FILE_NAME = "model.safetensors"
TEST_EXPECTED_HASH = "abc123"
TEST_DESTINATION_PATH = "/path/to/model.safetensors"


def create_test_model(session, file_name, model_type, hash_value, file_size=1000, source_url=None):
    """Helper to create a test model in the database."""
    model = Model(path=file_name, type=model_type, hash=hash_value, file_size=file_size, source_url=source_url)
    session.add(model)
    session.commit()
    return model


def setup_mock_hash_calculation(model_processor, hash_value):
    """Helper to setup hash calculation mocks."""
    mock_hash = MagicMock()
    mock_hash.hexdigest.return_value = hash_value
    return patch.object(model_processor, "_get_hasher", return_value=mock_hash)


def verify_model_in_db(session, file_name, expected_hash=None, expected_type=None):
    """Helper to verify model exists in database with correct attributes."""
    db_model = session.query(Model).filter_by(path=file_name).first()
    assert db_model is not None
    if expected_hash:
        assert db_model.hash == expected_hash
    if expected_type:
        assert db_model.type == expected_type
    return db_model


@pytest.fixture
def db_engine():
    # Configure in-memory database
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(db_engine):
    Session = sessionmaker(bind=db_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mock_get_relative_path():
    with patch("app.model_processor.get_relative_path") as mock:
        mock.side_effect = lambda path: (TEST_MODEL_TYPE, os.path.basename(path))
        yield mock


@pytest.fixture
def mock_get_full_path():
    with patch("app.model_processor.get_full_path") as mock:
        mock.return_value = TEST_DESTINATION_PATH
        yield mock


@pytest.fixture
def model_processor(db_session, mock_get_relative_path, mock_get_full_path):
    with patch("app.model_processor.create_session", return_value=db_session):
        with patch("app.model_processor.can_create_session", return_value=True):
            processor = ModelProcessor()
            # Setup test state
            processor.removed_files = []
            processor.downloaded_files = []
            processor.file_exists = {}

            def mock_download_file(url, destination_path, hasher):
                processor.downloaded_files.append((url, destination_path))
                processor.file_exists[destination_path] = True
                # Simulate writing some data to the file
                test_data = b"test data"
                hasher.update(test_data)

            def mock_remove_file(file_path):
                processor.removed_files.append(file_path)
                if file_path in processor.file_exists:
                    del processor.file_exists[file_path]

            # Setup common patches
            file_exists_patch = patch.object(
                processor,
                "_file_exists",
                side_effect=lambda path: processor.file_exists.get(path, False),
            )
            file_size_patch = patch.object(
                processor,
                "_get_file_size",
                side_effect=lambda path: (
                    1000 if processor.file_exists.get(path, False) else 0
                ),
            )
            download_file_patch = patch.object(
                processor, "_download_file", side_effect=mock_download_file
            )
            remove_file_patch = patch.object(
                processor, "_remove_file", side_effect=mock_remove_file
            )

            with (
                file_exists_patch,
                file_size_patch,
                download_file_patch,
                remove_file_patch,
            ):
                yield processor


def test_ensure_downloaded_invalid_extension(model_processor):
    # Ensure that an unsupported file extension raises an error to prevent unsafe file downloads
    with pytest.raises(ValueError, match="Unsupported unsafe file for download"):
        model_processor.ensure_downloaded(TEST_MODEL_TYPE, TEST_URL, "model.exe")


def test_ensure_downloaded_existing_file_with_hash(model_processor, db_session):
    # Ensure that a file with the same hash but from a different source is not downloaded again
    SOURCE_URL = "https://example.com/other.sft"
    create_test_model(db_session, TEST_FILE_NAME, TEST_MODEL_TYPE, TEST_EXPECTED_HASH, source_url=SOURCE_URL)
    model_processor.file_exists[TEST_DESTINATION_PATH] = True

    result = model_processor.ensure_downloaded(
        TEST_MODEL_TYPE, TEST_URL, TEST_FILE_NAME, TEST_EXPECTED_HASH
    )

    assert result == TEST_DESTINATION_PATH
    model = verify_model_in_db(db_session, TEST_FILE_NAME, TEST_EXPECTED_HASH, TEST_MODEL_TYPE)
    assert model.source_url == SOURCE_URL # Ensure the source URL is not overwritten


def test_ensure_downloaded_existing_file_hash_mismatch(model_processor, db_session):
    # Ensure that a file with a different hash raises an error
    create_test_model(db_session, TEST_FILE_NAME, TEST_MODEL_TYPE, "different_hash")
    model_processor.file_exists[TEST_DESTINATION_PATH] = True

    with pytest.raises(ValueError, match="File .* exists with hash .* but expected .*"):
        model_processor.ensure_downloaded(
            TEST_MODEL_TYPE, TEST_URL, TEST_FILE_NAME, TEST_EXPECTED_HASH
        )


def test_ensure_downloaded_new_file(model_processor, db_session):
    # Ensure that a new file is downloaded
    model_processor.file_exists[TEST_DESTINATION_PATH] = False

    with setup_mock_hash_calculation(model_processor, TEST_EXPECTED_HASH):
        result = model_processor.ensure_downloaded(
            TEST_MODEL_TYPE, TEST_URL, TEST_FILE_NAME, TEST_EXPECTED_HASH
        )

    assert result == TEST_DESTINATION_PATH
    assert len(model_processor.downloaded_files) == 1
    assert model_processor.downloaded_files[0] == (TEST_URL, TEST_DESTINATION_PATH)
    assert model_processor.file_exists[TEST_DESTINATION_PATH]
    verify_model_in_db(db_session, TEST_FILE_NAME, TEST_EXPECTED_HASH, TEST_MODEL_TYPE)


def test_ensure_downloaded_hash_mismatch(model_processor, db_session):
    # Ensure that download that results in a different hash raises an error
    model_processor.file_exists[TEST_DESTINATION_PATH] = False

    with setup_mock_hash_calculation(model_processor, "different_hash"):
        with pytest.raises(
            ValueError,
            match="Downloaded file hash .* does not match expected hash .*",
        ):
            model_processor.ensure_downloaded(
                TEST_MODEL_TYPE,
                TEST_URL,
                TEST_FILE_NAME,
                TEST_EXPECTED_HASH,
            )

    assert len(model_processor.removed_files) == 1
    assert model_processor.removed_files[0] == TEST_DESTINATION_PATH
    assert TEST_DESTINATION_PATH not in model_processor.file_exists
    assert db_session.query(Model).filter_by(path=TEST_FILE_NAME).first() is None


def test_process_file_without_hash(model_processor, db_session):
    # Test processing file without provided hash
    model_processor.file_exists[TEST_DESTINATION_PATH] = True

    with patch.object(model_processor, "_hash_file", return_value=TEST_EXPECTED_HASH):
        result = model_processor.process_file(TEST_DESTINATION_PATH)
        assert result is not None
        assert result.hash == TEST_EXPECTED_HASH


def test_retrieve_model_by_hash(model_processor, db_session):
    # Test retrieving model by hash
    create_test_model(db_session, TEST_FILE_NAME, TEST_MODEL_TYPE, TEST_EXPECTED_HASH)
    result = model_processor.retrieve_model_by_hash(TEST_EXPECTED_HASH)
    assert result is not None
    assert result.hash == TEST_EXPECTED_HASH


def test_retrieve_model_by_hash_and_type(model_processor, db_session):
    # Test retrieving model by hash and type
    create_test_model(db_session, TEST_FILE_NAME, TEST_MODEL_TYPE, TEST_EXPECTED_HASH)
    result = model_processor.retrieve_model_by_hash(TEST_EXPECTED_HASH, TEST_MODEL_TYPE)
    assert result is not None
    assert result.hash == TEST_EXPECTED_HASH
    assert result.type == TEST_MODEL_TYPE


def test_retrieve_hash(model_processor, db_session):
    # Test retrieving hash for existing model
    create_test_model(db_session, TEST_FILE_NAME, TEST_MODEL_TYPE, TEST_EXPECTED_HASH)
    with patch.object(
        model_processor,
        "_validate_path",
        return_value=(TEST_MODEL_TYPE, TEST_FILE_NAME),
    ):
        result = model_processor.retrieve_hash(TEST_DESTINATION_PATH, TEST_MODEL_TYPE)
        assert result == TEST_EXPECTED_HASH


def test_validate_file_extension_valid_extensions(model_processor):
    # Test all valid file extensions
    valid_extensions = [".safetensors", ".sft", ".txt", ".csv", ".json", ".yaml"]
    for ext in valid_extensions:
        model_processor._validate_file_extension(f"test{ext}")  # Should not raise


def test_process_file_existing_without_source_url(model_processor, db_session):
    # Test processing an existing file that needs its source URL updated
    model_processor.file_exists[TEST_DESTINATION_PATH] = True

    create_test_model(db_session, TEST_FILE_NAME, TEST_MODEL_TYPE, TEST_EXPECTED_HASH)
    result = model_processor.process_file(TEST_DESTINATION_PATH, source_url=TEST_URL)

    assert result is not None
    assert result.hash == TEST_EXPECTED_HASH
    assert result.source_url == TEST_URL

    db_model = db_session.query(Model).filter_by(path=TEST_FILE_NAME).first()
    assert db_model.source_url == TEST_URL
