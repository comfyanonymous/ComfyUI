"""init

Revision ID: 565b08122d00
Revises: 
Create Date: 2025-05-29 19:15:56.230322

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '565b08122d00'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('model',
    sa.Column('type', sa.Text(), nullable=False),
    sa.Column('path', sa.Text(), nullable=False),
    sa.Column('hash', sa.Text(), nullable=True),
    sa.Column('date_added', sa.DateTime(), server_default=sa.text('(CURRENT_TIMESTAMP)'), nullable=True),
    sa.PrimaryKeyConstraint('type', 'path')
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('model')
