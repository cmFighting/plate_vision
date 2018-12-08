# encoding: utf-8
import cv2
import flask
from flask import Flask, render_template, request, session, jsonify
from untils import *
from example.demo_sms_send import codesend
import config
from exts import db
from models import User, Station, Car, Records
import os
from decorate import login_required
import shortuuid
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)


# 主页
@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/index')
def index():
    return flask.render_template('index.html')


# 登录
# 登录函数, 输入手机号码,登录之后进行跳转
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        phone = request.form.get('phone')
        rawpassword = request.form.get('password')
        password = getmd5(phone, rawpassword)
        user = User.query.filter(User.phone==phone, User.password == password).first()
        print(user)
        if user:
            flask.session['id'] = user.uid
            flask.g.user = user
            # 因为采用的是异步刷新,所以不能携带参数进行返回,目前的方法,只能采用这个ajax跳转
            return jsonify({'result': 'ok'})
        else:
            return jsonify({'result': 'error'})


# 发送验证码
# 用户已经注册 返回repeat
# 验证码发送成功 返回 ok, 失败返回error
@app.route('/sendcode')
def sendcode():
    phone = request.args.get('phone')
    phone = str(phone)
    user = User.query.filter(User.phone == phone).first()
    if user:
        return jsonify({'result': 'repeat'})
    else:
        code = get6code()
        result = codesend(phone=phone, code=code)
        # print(result)
        if result == "OK":
            # flask.g.code = code
            flask.session['code'] = code
            return jsonify({'result': 'ok'})
        else:
            return jsonify({'result': 'error'})
    #下面这部分是调用手机号将验证码发送给用户


# 注册
@app.route('/regist', methods=['GET', 'POST'])
def regist():
    if request.method == 'GET':
        return render_template('regist.html')
    else:
        telphone = request.form.get('phone')
        rawpassword = request.form.get('password')
        username = request.form.get('username')
        code = request.form.get('code')
        print(code, username, rawpassword, telphone)
        print(flask.session['code'])
        if(code != flask.session['code']):
            print("这边是返回错误")
            return jsonify({'result': 'codeerror'})
        else:
            password = getmd5(telphone, rawpassword)
            print(password)
            # 发送验证码的时候用户已经检查了重复
            user = User(username=username, phone=telphone, password=password)
            db.session.add(user)
            db.session.commit()
            return jsonify({'result': 'ok'})


# 清除session并退出账户
@app.route('/logout')
def logout():
    flask.session.clear()
    return flask.redirect(flask.url_for('login'))


@app.route('/bind', methods=['POST', 'GET'])  # 添加路由 ,准确来说,绑定车牌信息,相当于是一个更新的操作,通过更新这个操作将车牌信息进行绑定
@login_required
def bind():
    if request.method == 'GET':
        return flask.render_template('bind.html')
    else:
        plate = request.form.get('plate')
        car = Car.query.filter(Car.platenumber == plate).first()
        if car:
            car.user_id = flask.session['id']
            db.session.commit()
            # 提示绑定成功.或者是返回详情页面,或者是主页
            # 这边绑定车牌信息成功之后你要进行返回
            print("绑定车牌信息成功!")
            jsonify({'result': 'ok'})
        else:
            return jsonify({'result': 'not_found'})


# @login_required
# @app.route('/myinfo')
# def myinfo():
#     # 以列表的形式传递进来,之后将字典值打散传入
#     records = Car.query.filter(Records.platenumber)
    # 向数据库发起查询,显示列表信息, 通过一个循环遍历的形式\


# 检查是否有session, 判断用户登录信息
@app.before_request
def before_request():
    id = flask.session.get('id')
    if id:
        user = User.query.get(id)
        flask.g.user = user


# 供所有的模板使用session的信息
@app.context_processor
def my_context_processor():
    if hasattr(flask.g, 'user'):
        return {"user": flask.g.user}
    else:
        return {}


if __name__ == '__main__':
    app.run()
