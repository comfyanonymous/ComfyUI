import os
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import select

from app.assets.database.models import Asset, AssetContent, AssetTag
from app.assets.scanner import (
    build_asset_specs,
    mark_contents_missing_outside_prefixes,
    seed_asset_specs,
    sync_prefixes_with_filesystem,
)


def _build_seed_specs(root: Path) -> list:
    return build_asset_specs(
        [str(path) for path in sorted(root.iterdir())],
        existing_paths=set(),
        enable_metadata_extraction=False,
    )[0]


def test_seed_creates_content_and_record(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    (input_root / "first.png").write_bytes(b"first")
    (input_root / "second.png").write_bytes(b"second")

    with patch("folder_paths.get_input_directory", return_value=str(input_root)):
        created = seed_asset_specs(session, _build_seed_specs(input_root))
    session.commit()

    contents = list(session.scalars(select(AssetContent).order_by(AssetContent.path)))
    records = list(session.scalars(select(Asset).order_by(Asset.name)))

    assert created == 2
    assert len(contents) == 2
    assert len(records) == 2
    assert [record.loader_path for record in records] == ["first.png", "second.png"]
    assert [link.tag_name for link in session.scalars(select(AssetTag).order_by(AssetTag.asset_id))] == [
        "input",
        "input",
    ]


def test_prune_marks_missing_not_deletes(session, temp_dir: Path):
    input_root = temp_dir / "input"
    input_root.mkdir()
    file_path = input_root / "removed-from-registry.png"
    file_path.write_bytes(b"content")

    with patch("folder_paths.get_input_directory", return_value=str(input_root)):
        seed_asset_specs(session, _build_seed_specs(input_root))
    session.commit()

    marked = mark_contents_missing_outside_prefixes(session, prefixes=[])
    session.commit()

    content = session.scalar(select(AssetContent))
    record = session.scalar(select(Asset))
    missing_tag = session.get(AssetTag, {"asset_id": record.id, "tag_name": "missing"})

    assert marked == 1
    assert content is not None and content.is_missing is True
    assert record is not None
    assert missing_tag is not None and missing_tag.origin == "automatic"


def test_unhashed_missing_content_gets_tagged(session, temp_dir: Path):
    missing_path = os.path.abspath(temp_dir / "missing.bin")
    content = AssetContent(path=missing_path, hash=None, size_bytes=7, mtime_ns=1)
    session.add(content)
    session.flush()
    record = Asset(content_id=content.id, name="missing.bin")
    session.add(record)
    session.commit()

    sync_prefixes_with_filesystem(session, prefixes=[str(temp_dir)])
    session.commit()

    session.expire_all()
    content = session.get(AssetContent, content.id)
    missing_tag = session.get(AssetTag, {"asset_id": record.id, "tag_name": "missing"})

    assert content is not None and content.is_missing is True
    assert missing_tag is not None and missing_tag.origin == "automatic"
