"""init tables

Revision ID: 388bd1c104f2
Revises: 
Create Date: 2024-12-11 20:24:41.021041

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '388bd1c104f2'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None





def upgrade() -> None:
    op.execute("""
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS user_movie (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    movie_id UUID NOT NULL
);
               
ALTER TABLE user_movie
ADD CONSTRAINT unique_user_movie_pair UNIQUE (user_id, movie_id);

""")


def downgrade() -> None:
    op.execute("DROP TABLE user_movie;")
