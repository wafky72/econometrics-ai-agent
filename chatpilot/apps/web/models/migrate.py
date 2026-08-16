from peewee import *
from playhouse.migrate import *
from chatpilot.apps.db import DB

# 创建迁移器
migrator = SqliteMigrator(DB)

# 执行迁移
with DB.atomic():
    migrate(

        migrator.add_column('user', 'uploaded_files', TextField(default='[]'))

    )