"""Adding a new value to an Enum is not handled by alembic.

Revision ID: manual2
Revises: manual1
Create Date: 2017-05-06 16:41:04.936361

"""

# revision identifiers, used by Alembic.
revision = 'manual2'
down_revision = 'manual1'

from alembic import op
import sqlalchemy as sa


name = 'user_interaction_type'
tmp_name = 'tmp_' + name

old_options = (
    'copy',
    'download',
    'giving credit',
    'landing',
    'login',
    'logout',
    'looking at error',
    'looking at event',
    'looking at leaderboard',
    'looking at my_submissions',
    'looking at private leaderboard',
    'looking at submission',
    'looking at user',
    'save',
    'signing up at event',
    'submit',
    'upload',
)

new_options = sorted(old_options + ('looking at problems',))

new_type = sa.Enum(*new_options, name=name)
old_type = sa.Enum(*old_options, name=name)

table_1 = 'user_interactions'
column = 'interaction'  # change also manually in downgrade

tcr_1 = sa.sql.table(
    table_1, sa.Column(column, new_type))


def upgrade():
    op.execute('ALTER TYPE ' + name + ' RENAME TO ' + tmp_name)
    new_type.create(op.get_bind())
    op.execute('ALTER TABLE ' + table_1 + ' ALTER COLUMN ' + column +
               ' TYPE ' + name + ' USING ' + column + '::text::' + name)
    op.execute('DROP TYPE ' + tmp_name)


def downgrade():
    op.execute(tcr_1.update().where(tcr_1.c.state == 'looking at problems')
               .values(state='looking at event'))
    op.execute('ALTER TYPE ' + name + ' RENAME TO ' + tmp_name)
    old_type.create(op.get_bind())
    op.execute('ALTER TABLE ' + table_1 + ' ALTER COLUMN ' + column +
               ' TYPE ' + name + ' USING ' + column + '::text::' + name)
    op.execute('DROP TYPE ' + tmp_name)
