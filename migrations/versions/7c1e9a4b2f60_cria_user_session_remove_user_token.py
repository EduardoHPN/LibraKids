"""cria user_session e remove user.token

Os tokens atuais deixam de valer: os usuários precisam entrar de novo. As contas
e os hashes de senha não mudam. O downgrade recria a coluna vazia e apaga as
sessões, então os tokens não podem ser recuperados.

Revision ID: 7c1e9a4b2f60
Revises: 4dc8d8585ded
Create Date: 2026-11-04 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7c1e9a4b2f60'
down_revision = '4dc8d8585ded'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('user_session',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('token_hash', sa.String(length=64), nullable=False),
    sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name=op.f('fk_user_session_user_id_user'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_user_session')),
    sa.UniqueConstraint('token_hash', name=op.f('uq_user_session_token_hash'))
    )
    op.create_index(op.f('ix_user_session_user_id'), 'user_session', ['user_id'], unique=False)

    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f('uq_user_token'), type_='unique')
        batch_op.drop_column('token')


def downgrade():
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('token', sa.String(length=64), nullable=True))
        batch_op.create_unique_constraint(batch_op.f('uq_user_token'), ['token'])

    op.drop_index(op.f('ix_user_session_user_id'), table_name='user_session')
    op.drop_table('user_session')
