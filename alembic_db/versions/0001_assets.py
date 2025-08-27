# File: /alembic_db/versions/0001_assets.py
"""initial assets schema + per-asset state cache

Revision ID: 0001_assets
Revises:
Create Date: 2025-08-20 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "0001_assets"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ASSETS: content identity (deduplicated by hash)
    op.create_table(
        "assets",
        sa.Column("hash", sa.String(length=256), primary_key=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.CheckConstraint("size_bytes >= 0", name="ck_assets_size_nonneg"),
    )
    op.create_index("ix_assets_mime_type", "assets", ["mime_type"])

    # ASSETS_INFO: user-visible references (mutable metadata)
    op.create_table(
        "assets_info",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("asset_hash", sa.String(length=256), sa.ForeignKey("assets.hash", ondelete="RESTRICT"), nullable=False),
        sa.Column("preview_hash", sa.String(length=256), sa.ForeignKey("assets.hash", ondelete="SET NULL"), nullable=True),
        sa.Column("user_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("last_access_time", sa.DateTime(timezone=False), nullable=False),
        sa.UniqueConstraint("asset_hash", "owner_id", "name", name="uq_assets_info_hash_owner_name"),
        sqlite_autoincrement=True,
    )
    op.create_index("ix_assets_info_owner_id", "assets_info", ["owner_id"])
    op.create_index("ix_assets_info_asset_hash", "assets_info", ["asset_hash"])
    op.create_index("ix_assets_info_name", "assets_info", ["name"])
    op.create_index("ix_assets_info_created_at", "assets_info", ["created_at"])
    op.create_index("ix_assets_info_last_access_time", "assets_info", ["last_access_time"])
    op.create_index("ix_assets_info_owner_name", "assets_info", ["owner_id", "name"])

    # TAGS: normalized tag vocabulary
    op.create_table(
        "tags",
        sa.Column("name", sa.String(length=512), primary_key=True),
        sa.Column("tag_type", sa.String(length=32), nullable=False, server_default="user"),
        sa.CheckConstraint("name = lower(name)", name="ck_tags_lowercase"),
    )
    op.create_index("ix_tags_tag_type", "tags", ["tag_type"])

    # ASSET_INFO_TAGS: many-to-many for tags on AssetInfo
    op.create_table(
        "asset_info_tags",
        sa.Column("asset_info_id", sa.Integer(), sa.ForeignKey("assets_info.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tag_name", sa.String(length=512), sa.ForeignKey("tags.name", ondelete="RESTRICT"), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False, server_default="manual"),
        sa.Column("added_at", sa.DateTime(timezone=False), nullable=False),
        sa.PrimaryKeyConstraint("asset_info_id", "tag_name", name="pk_asset_info_tags"),
    )
    op.create_index("ix_asset_info_tags_tag_name", "asset_info_tags", ["tag_name"])
    op.create_index("ix_asset_info_tags_asset_info_id", "asset_info_tags", ["asset_info_id"])

    # ASSET_CACHE_STATE: 1:1 local cache metadata for an Asset
    op.create_table(
        "asset_cache_state",
        sa.Column("asset_hash", sa.String(length=256), sa.ForeignKey("assets.hash", ondelete="CASCADE"), primary_key=True),
        sa.Column("file_path", sa.Text(), nullable=False),  # absolute local path to cached file
        sa.Column("mtime_ns", sa.BigInteger(), nullable=True),
        sa.CheckConstraint("(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_acs_mtime_nonneg"),
    )
    op.create_index("ix_asset_cache_state_file_path", "asset_cache_state", ["file_path"])

    # ASSET_INFO_META: typed KV projection of user_metadata for filtering/sorting
    op.create_table(
        "asset_info_meta",
        sa.Column("asset_info_id", sa.Integer(), sa.ForeignKey("assets_info.id", ondelete="CASCADE"), nullable=False),
        sa.Column("key", sa.String(length=256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("val_str", sa.String(length=2048), nullable=True),
        sa.Column("val_num", sa.Numeric(38, 10), nullable=True),
        sa.Column("val_bool", sa.Boolean(), nullable=True),
        sa.Column("val_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("asset_info_id", "key", "ordinal", name="pk_asset_info_meta"),
    )
    op.create_index("ix_asset_info_meta_key", "asset_info_meta", ["key"])
    op.create_index("ix_asset_info_meta_key_val_str", "asset_info_meta", ["key", "val_str"])
    op.create_index("ix_asset_info_meta_key_val_num", "asset_info_meta", ["key", "val_num"])
    op.create_index("ix_asset_info_meta_key_val_bool", "asset_info_meta", ["key", "val_bool"])

    # ASSET_LOCATIONS: remote locations per asset
    op.create_table(
        "asset_locations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("asset_hash", sa.String(length=256), sa.ForeignKey("assets.hash", ondelete="CASCADE"), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),  # e.g., "gcs"
        sa.Column("locator", sa.Text(), nullable=False),             # e.g., "gs://bucket/path/to/blob"
        sa.Column("expected_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("etag", sa.String(length=256), nullable=True),
        sa.Column("last_modified", sa.String(length=128), nullable=True),
        sa.UniqueConstraint("asset_hash", "provider", "locator", name="uq_asset_locations_triplet"),
    )
    op.create_index("ix_asset_locations_hash", "asset_locations", ["asset_hash"])
    op.create_index("ix_asset_locations_provider", "asset_locations", ["provider"])

    # Tags vocabulary for models
    tags_table = sa.table(
        "tags",
        sa.column("name", sa.String(length=512)),
        sa.column("tag_type", sa.String()),
    )
    op.bulk_insert(
        tags_table,
        [
            # Root folder tags
            {"name": "models", "tag_type": "system"},
            {"name": "input", "tag_type": "system"},
            {"name": "output", "tag_type": "system"},

            # Core tags
            {"name": "configs", "tag_type": "system"},
            {"name": "checkpoints", "tag_type": "system"},
            {"name": "loras", "tag_type": "system"},
            {"name": "vae", "tag_type": "system"},
            {"name": "text_encoders", "tag_type": "system"},
            {"name": "diffusion_models", "tag_type": "system"},
            {"name": "clip_vision", "tag_type": "system"},
            {"name": "style_models", "tag_type": "system"},
            {"name": "embeddings", "tag_type": "system"},
            {"name": "diffusers", "tag_type": "system"},
            {"name": "vae_approx", "tag_type": "system"},
            {"name": "controlnet", "tag_type": "system"},
            {"name": "gligen", "tag_type": "system"},
            {"name": "upscale_models", "tag_type": "system"},
            {"name": "hypernetworks", "tag_type": "system"},
            {"name": "photomaker", "tag_type": "system"},
            {"name": "classifiers", "tag_type": "system"},

            # Extra basic tags (used for vae_approx, ...)
            {"name": "encoder", "tag_type": "system"},
            {"name": "decoder", "tag_type": "system"},
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_asset_locations_provider", table_name="asset_locations")
    op.drop_index("ix_asset_locations_hash", table_name="asset_locations")
    op.drop_table("asset_locations")

    op.drop_index("ix_asset_info_meta_key_val_bool", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_num", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_str", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key", table_name="asset_info_meta")
    op.drop_table("asset_info_meta")

    op.drop_index("ix_asset_cache_state_file_path", table_name="asset_cache_state")
    op.drop_table("asset_cache_state")

    op.drop_index("ix_asset_info_tags_asset_info_id", table_name="asset_info_tags")
    op.drop_index("ix_asset_info_tags_tag_name", table_name="asset_info_tags")
    op.drop_table("asset_info_tags")

    op.drop_index("ix_tags_tag_type", table_name="tags")
    op.drop_table("tags")

    op.drop_constraint("uq_assets_info_hash_owner_name", table_name="assets_info")
    op.drop_index("ix_assets_info_owner_name", table_name="assets_info")
    op.drop_index("ix_assets_info_last_access_time", table_name="assets_info")
    op.drop_index("ix_assets_info_created_at", table_name="assets_info")
    op.drop_index("ix_assets_info_name", table_name="assets_info")
    op.drop_index("ix_assets_info_asset_hash", table_name="assets_info")
    op.drop_index("ix_assets_info_owner_id", table_name="assets_info")
    op.drop_table("assets_info")

    op.drop_index("ix_assets_mime_type", table_name="assets")
    op.drop_table("assets")
