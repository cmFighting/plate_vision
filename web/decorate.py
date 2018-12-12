from functools import wraps
import flask


# 登录装饰器
# 如果用户没有登录的话.会跳转到登录页面,是不能直接进行调用的
def login_required(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        if hasattr(flask.g, 'user'):
            return func(*args, **kargs)
        else:
            return flask.redirect(flask.url_for('login'))

    return wrapper

