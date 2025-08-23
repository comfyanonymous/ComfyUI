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
        sa.Column("hash", sa.String(length=128), primary_key=True),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("mime_type", sa.String(length=255), nullable=True),
        sa.Column("refcount", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("storage_backend", sa.String(length=32), nullable=False, server_default="fs"),
        sa.Column("storage_locator", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.CheckConstraint("size_bytes >= 0", name="ck_assets_size_nonneg"),
        sa.CheckConstraint("refcount >= 0", name="ck_assets_refcount_nonneg"),
    )
    op.create_index("ix_assets_mime_type", "assets", ["mime_type"])
    op.create_index("ix_assets_backend_locator", "assets", ["storage_backend", "storage_locator"])

    # ASSETS_INFO: user-visible references (mutable metadata)
    op.create_table(
        "assets_info",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_id", sa.String(length=128), nullable=True),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column("asset_hash", sa.String(length=128), sa.ForeignKey("assets.hash", ondelete="RESTRICT"), nullable=False),
        sa.Column("preview_hash", sa.String(length=128), sa.ForeignKey("assets.hash", ondelete="SET NULL"), nullable=True),
        sa.Column("user_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_access_time", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sqlite_autoincrement=True,
    )
    op.create_index("ix_assets_info_owner_id", "assets_info", ["owner_id"])
    op.create_index("ix_assets_info_asset_hash", "assets_info", ["asset_hash"])
    op.create_index("ix_assets_info_name", "assets_info", ["name"])
    op.create_index("ix_assets_info_created_at", "assets_info", ["created_at"])
    op.create_index("ix_assets_info_last_access_time", "assets_info", ["last_access_time"])

    # TAGS: normalized tag vocabulary
    op.create_table(
        "tags",
        sa.Column("name", sa.String(length=128), primary_key=True),
        sa.Column("tag_type", sa.String(length=32), nullable=False, server_default="user"),
        sa.CheckConstraint("name = lower(name)", name="ck_tags_lowercase"),
    )
    op.create_index("ix_tags_tag_type", "tags", ["tag_type"])

    # ASSET_INFO_TAGS: many-to-many for tags on AssetInfo
    op.create_table(
        "asset_info_tags",
        sa.Column("asset_info_id", sa.BigInteger(), sa.ForeignKey("assets_info.id", ondelete="CASCADE"), nullable=False),
        sa.Column("tag_name", sa.String(length=128), sa.ForeignKey("tags.name", ondelete="RESTRICT"), nullable=False),
        sa.Column("origin", sa.String(length=32), nullable=False, server_default="manual"),
        sa.Column("added_by", sa.String(length=128), nullable=True),
        sa.Column("added_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("asset_info_id", "tag_name", name="pk_asset_info_tags"),
    )
    op.create_index("ix_asset_info_tags_tag_name", "asset_info_tags", ["tag_name"])
    op.create_index("ix_asset_info_tags_asset_info_id", "asset_info_tags", ["asset_info_id"])

    # ASSET_LOCATOR_STATE: 1:1 filesystem metadata(for fast integrity checking) for an Asset records
    op.create_table(
        "asset_locator_state",
        sa.Column("asset_hash", sa.String(length=128), sa.ForeignKey("assets.hash", ondelete="CASCADE"), primary_key=True),
        sa.Column("mtime_ns", sa.BigInteger(), nullable=True),
        sa.Column("etag", sa.String(length=256), nullable=True),
        sa.Column("last_modified", sa.String(length=128), nullable=True),
        sa.CheckConstraint("(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_als_mtime_nonneg"),
    )

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

    # Tags vocabulary for models
    tags_table = sa.table(
        "tags",
        sa.column("name", sa.String()),
        sa.column("tag_type", sa.String()),
    )
    op.bulk_insert(
        tags_table,
        [
            # Core concept tags
            {"name": "models", "tag_type": "system"},

            # Canonical single-word types
            {"name": "checkpoint", "tag_type": "system"},
            {"name": "lora", "tag_type": "system"},
            {"name": "vae", "tag_type": "system"},
            {"name": "text-encoder", "tag_type": "system"},
            {"name": "clip-vision", "tag_type": "system"},
            {"name": "embedding", "tag_type": "system"},
            {"name": "controlnet", "tag_type": "system"},
            {"name": "upscale", "tag_type": "system"},
            {"name": "diffusion-model", "tag_type": "system"},
            {"name": "hypernetwork", "tag_type": "system"},
            {"name": "vae-approx", "tag_type": "system"},
            {"name": "gligen", "tag_type": "system"},
            {"name": "style-model", "tag_type": "system"},
            {"name": "encoder", "tag_type": "system"},
            {"name": "decoder", "tag_type": "system"},
            # TODO: decide what to do with: photomaker, classifiers
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_asset_info_meta_key_val_bool", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_num", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_str", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key", table_name="asset_info_meta")
    op.drop_table("asset_info_meta")

    op.drop_table("asset_locator_state")

    op.drop_index("ix_asset_info_tags_asset_info_id", table_name="asset_info_tags")
    op.drop_index("ix_asset_info_tags_tag_name", table_name="asset_info_tags")
    op.drop_table("asset_info_tags")

    op.drop_index("ix_tags_tag_type", table_name="tags")
    op.drop_table("tags")

    op.drop_index("ix_assets_info_last_access_time", table_name="assets_info")
    op.drop_index("ix_assets_info_created_at", table_name="assets_info")
    op.drop_index("ix_assets_info_name", table_name="assets_info")
    op.drop_index("ix_assets_info_asset_hash", table_name="assets_info")
    op.drop_index("ix_assets_info_owner_id", table_name="assets_info")
    op.drop_table("assets_info")

    op.drop_index("ix_assets_backend_locator", table_name="assets")
    op.drop_index("ix_assets_mime_type", table_name="assets")
    op.drop_table("assets")
