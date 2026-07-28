"""
Download manager schema.

Adds the two tables that back the server-side model download manager:
transient job/queue state (``downloads`` + per-segment ``download_segments``).

Revision ID: 0005_download_manager
Revises: 0004_drop_tag_type
Create Date: 2026-06-27
"""

from alembic import op
import sqlalchemy as sa

revision = "0005_download_manager"
down_revision = "0004_drop_tag_type"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "downloads",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("final_url", sa.Text(), nullable=True),
        sa.Column("model_id", sa.String(length=1024), nullable=False),
        sa.Column("dest_path", sa.Text(), nullable=False),
        sa.Column("temp_path", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=True),
        sa.Column("bytes_done", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("etag", sa.String(length=512), nullable=True),
        sa.Column("last_modified", sa.String(length=128), nullable=True),
        sa.Column(
            "accept_ranges", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("expected_sha256", sa.String(length=64), nullable=True),
        sa.Column(
            "allow_any_extension",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.BigInteger(), nullable=False),
        sa.Column("updated_at", sa.BigInteger(), nullable=False),
        sa.CheckConstraint("bytes_done >= 0", name="ck_downloads_bytes_done_nonneg"),
        sa.CheckConstraint(
            "total_bytes IS NULL OR total_bytes >= 0",
            name="ck_downloads_total_bytes_nonneg",
        ),
    )
    op.create_index("ix_downloads_status", "downloads", ["status"])
    op.create_index("ix_downloads_priority", "downloads", ["priority"])
    op.create_index("ix_downloads_model_id", "downloads", ["model_id"])

    op.create_table(
        "download_segments",
        sa.Column(
            "download_id",
            sa.String(length=36),
            sa.ForeignKey("downloads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("start_offset", sa.BigInteger(), nullable=False),
        sa.Column("end_offset", sa.BigInteger(), nullable=False),
        sa.Column("bytes_done", sa.BigInteger(), nullable=False, server_default="0"),
        sa.PrimaryKeyConstraint("download_id", "idx", name="pk_download_segments"),
        sa.CheckConstraint("bytes_done >= 0", name="ck_segments_bytes_done_nonneg"),
        sa.CheckConstraint("end_offset >= start_offset", name="ck_segments_range"),
    )


def downgrade() -> None:
    op.drop_table("download_segments")

    op.drop_index("ix_downloads_model_id", table_name="downloads")
    op.drop_index("ix_downloads_priority", table_name="downloads")
    op.drop_index("ix_downloads_status", table_name="downloads")
    op.drop_table("downloads")
