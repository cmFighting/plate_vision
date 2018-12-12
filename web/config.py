# encoding: utf-8
# 项目配置文件
import os

# 数据库配置
DIALECT = 'mysql'
DRIVER = 'pymysql'
USERNAME = 'root'
PASSWORD = 'password'
HOST = '120.79.173.196'
PORT = '3306'
DATABASE = 'luffy'
SQLALCHEMY_DATABASE_URI = "{}+{}://{}:{}@{}:{}/{}?charset=utf8".format(DIALECT, DRIVER, USERNAME, PASSWORD, HOST, PORT, DATABASE)
SQLALCHEMY_TRACK_MODIFICATIONS = True
# debug模式
DEBUG = True
# 用于session的key
SECRET_KEY = os.urandom(24)