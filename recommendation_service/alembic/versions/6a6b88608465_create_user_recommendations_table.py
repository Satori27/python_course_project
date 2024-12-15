"""Create user_recommendations table

Revision ID: 6a6b88608465
Revises: 
Create Date: 2024-12-09 16:37:06.400483

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '6a6b88608465'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE TABLE If NOT EXISTS user_recommendations (
        id SERIAL PRIMARY KEY,
        user_id INT NOT NULL,
        movie_id INT NOT NULL
);
""")


def downgrade() -> None:
    op.execute("DROP TABLE user_recommendations;")
