"""Adding a new value to an Enum is not handled by alembic.

Revision ID: manual1
Revises: 4211b7c3e1e8
Create Date: 2017-01-13 20:13:04.936361

"""

# revision identifiers, used by Alembic.
revision = 'manual1'
down_revision = '4211b7c3e1e8'

from alembic import op
import sqlalchemy as sa


name = 'submission_states'
tmp_name = 'tmp_' + name

old_options = (
    'new', 'checked', 'checking_error', 'trained', 'training_error',
    'validated', 'validating_error', 'tested', 'testing_error')
new_options = sorted(old_options + ('training',))

new_type = sa.Enum(*new_options, name=name)
old_type = sa.Enum(*old_options, name=name)

table_1 = 'submissions'
table_2 = 'submission_on_cv_folds'
column = 'state'  # change also manually in downgrade

tcr_1 = sa.sql.table(
    table_1, sa.Column(column, new_type, default='new'))
tcr_2 = sa.sql.table(
    table_2, sa.Column(column, new_type, default='new'))


def upgrade():
    op.execute('ALTER TYPE ' + name + ' RENAME TO ' + tmp_name)
    new_type.create(op.get_bind())
    op.execute('ALTER TABLE ' + table_1 + ' ALTER COLUMN ' + column +
               ' TYPE ' + name + ' USING ' + column + '::text::' + name)
    op.execute('ALTER TABLE ' + table_2 + ' ALTER COLUMN ' + column +
               ' TYPE ' + name + ' USING ' + column + '::text::' + name)
    op.execute('DROP TYPE ' + tmp_name)


def downgrade():
    op.execute(tcr_1.update().where(tcr_1.c.state=='training')
               .values(state='new'))
    op.execute(tcr_2.update().where(tcr_2.c.state=='training')
               .values(state='new'))
    op.execute('ALTER TYPE ' + name + ' RENAME TO ' + tmp_name)
    old_type.create(op.get_bind())
    op.execute('ALTER TABLE ' + table_1 + ' ALTER COLUMN ' + column +
               ' TYPE ' + name + ' USING ' + column + '::text::' + name)
    op.execute('ALTER TABLE ' + table_2 + ' ALTER COLUMN ' + column +
               ' TYPE ' + name + ' USING ' + column + '::text::' + name)
    op.execute('DROP TYPE ' + tmp_name)