"""create file_types table

Revision ID: 55206be47b71
Revises: 
Create Date: 2015-12-09 22:22:28.923933

"""

# revision identifiers, used by Alembic.
revision = '55206be47b71'
down_revision = None
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from databoard import db
import databoard.db_tools as db_tools
from databoard.model import FileType, SubmissionFile
import databoard.ramps.air_passengers.specific as specific


def upgrade():
    op.create_table(
        'file_types',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('is_editable', sa.Boolean(), nullable=True),
        sa.Column('max_size', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.add_column(u'submission_files', sa.Column('file_type_id', sa.Integer(),
                  nullable=True))
    db_tools.setup_problem(specific.file_types)
    file_types = FileType.query.all()
    file_types_dict = dict(
        [(file_type.name, file_type) for file_type in file_types])
    submission_files = SubmissionFile.query.all()
    for submission_file in submission_files:
        submission_file.file_type = file_types_dict[submission_file.name]
    db.session.commit()


def downgrade():
    pass