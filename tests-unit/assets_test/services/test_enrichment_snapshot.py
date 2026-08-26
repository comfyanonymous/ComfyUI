from pathlib import Path
from unittest.mock import patch

from app.assets.database.models import Asset, AssetContent
from app.assets.helpers import to_stored_hash
from app.assets.scanner import enrich_asset


def _create_unhashed_record(session, path: Path) -> tuple[AssetContent, Asset]:
    content = AssetContent(
        path=str(path),
        hash=None,
        size_bytes=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
    )
    session.add(content)
    session.flush()
    record = Asset(content_id=content.id, name=path.name)
    session.add(record)
    session.commit()
    return content, record


def test_enrichment_uses_snapshot_hash_not_direct_blake3(session, temp_dir: Path):
    path = temp_dir / "stable.bin"
    path.write_bytes(b"stable content")
    content, record = _create_unhashed_record(session, path)

    with patch(
        "app.assets.scanner.snapshot_hash", return_value="snapshot-digest"
    ) as mocked_snapshot_hash:
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=False,
            compute_hash=True,
        )

    assert enriched is True
    assert session.get(AssetContent, content.id).hash == to_stored_hash("snapshot-digest")
    mocked_snapshot_hash.assert_called_once_with(str(path))


def test_enrichment_discards_unstable_hash(session, temp_dir: Path):
    path = temp_dir / "unstable.bin"
    path.write_bytes(b"unstable content")
    content, record = _create_unhashed_record(session, path)

    with patch("app.assets.scanner.snapshot_hash", return_value=None):
        enriched = enrich_asset(
            session,
            file_path=str(path),
            content_id=content.id,
            record_id=record.id,
            extract_metadata=False,
            compute_hash=True,
        )

    assert enriched is False
    assert session.get(AssetContent, content.id).hash is None
