# encoding: utf-8
from exts import db
import shortuuid


# 用户表
class User(db.Model):
    __tablename__ = 'user'
    uid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)


# 站点表
class Station(db.Model):
    __tablename__ = 'station'
    station_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    station_name = db.Column(db.String(255), nullable=False)
    station_status = db.Column(db.String(255), nullable=False, default='1')


# 车辆信息表
class Car(db.Model):
    __tablename__ = 'car'
    platenumber = db.Column(db.String(255), primary_key=True)
    car_name = db.Column(db.String(255), nullable=False)
    car_type = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, nullable=True)


class Records(db.Model):
    __tablename__ = 'records'
    records_id = db.Column(db.String(255), primary_key=True, default=shortuuid.uuid())
    platenumber = db.Column(db.String(255), nullable=True)
    path_in_id = db.Column(db.Integer, nullable=True)
    path_in_time = db.Column(db.DateTime, nullable=True)
    path_out_id = db.Column(db.String(255), nullable=True)
    path_out_time = db.Column(db.DateTime, nullable=True)
