# encoding: utf-8
from flask_script import Manager
from flask_migrate import MigrateCommand, Migrate
from plate_app import app
from exts import db
import config
from models import User, Station, Car, Records  # 引入模型

app.config.from_object(config)
db.init_app(app)
manager = Manager(app)

# migrate绑定app和db
migrate = Migrate(app, db)

manager.add_command('db', MigrateCommand)

if __name__ == '__main__':
    manager.run()


