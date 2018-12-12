# 高速公路无感支付系统

这个一个Python课程实验项目，完整的实现了车牌识别、控制车辆放行、通过Web管理方式管理等功能

涉及到的技术：
1. `TensorFlow`训练中文省份简称以及数字字母
2. `Arduino`+超声波距离传感器+摄像头+控制电机
3. `Flask`后台管理

### 代码结构
```
.
├── lpr                                   # 车牌识别模块
│   ├── crop.jpg
│   ├── process                           # 车牌识别中的字符分割与识别模块
│   │   ├── characterSegmentation.py
│   │   ├── images
│   │   ├── __init__.py
│   │   ├── licensePlateLocation.py
│   │   └── __pycache__
│   ├── test.jpg
│   ├── test.py                            # 车牌识别的一个简单测试例子
│   └── train_model                        # 车牌识别中的训练模型模块
│       ├── cnn_chinese.py
│       ├── cnn_engLetter.py
│       ├── cnn_later.py
│       ├── __init__.py
│       ├── model
│       ├── __pycache__
│       ├── readmodel.py
│       └── train
├── readme.md
└── web                                    # web管理模块
    ├── aliyunsdkdysmsapi                  # 阿里云短信API的SDK
    │   ├── __init__.py
    │   └── request
    ├── config.py                          # flask应用的相关配置，例如数据库信息
    ├── decorate.py
    ├── example                            # 阿里云短信API的简易封装，方便在程序中调用
    │   └── demo_sms_send.py
    ├── exts.py
    ├── manage.py
    ├── migrations
    │   ├── alembic.ini
    │   ├── env.py
    │   ├── README
    │   ├── script.py.mako
    │   └── versions
    ├── models.py
    ├── plate_app.py                       # flask主程序文件
    ├── requirements.txt
    ├── static
    │   ├── css
    │   ├── images
    │   └── js
    ├── templates
    │   ├── base.html
    │   ├── bind.html
    │   ├── index.html
    │   ├── login.html
    │   ├── my_info.html
    │   └── regist.html
    └── untils.py
```

### 如何运行

##### 下载代码

* git clone https://github.com/cmFighting/plate_vision
* cd plate_vision

##### 运行web端

1. cd web
2. pip install -r requirements.txt # flask 相关依赖库
3. python plate_app.py

##### 训练模型

1. cd lpr
2. pip install -r requirements.txt # 安装tensorflow、numpy、opencv 相关依赖库
3. cd lpr/train_model
4. python cnn_chinese.py
5. python cnn_engLetter.py
6. python cnn_later.py

##### 运行车牌识别测试程序

1. cd lpr
3. python test.py

### 计划任务

- [x] 整理代码结构
- [ ] 完善相关文档资料
- [ ] 优化代码结构

### 参考学习链接

1. [tensorflow](https://www.tensorflow.org/)
2. [numpy](http://www.numpy.org/)
3. [arduino](https://www.arduino.cc/)
4. [face-recognition-with-python](https://realpython.com/face-recognition-with-python/)
5. [flask-sqlalchemy](http://flask-sqlalchemy.pocoo.org/2.3/quickstart/)
6. [flask](http://flask.pocoo.org/)

### 运行截图

**敬请期待**

### License

MIT