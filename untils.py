import hashlib
import random
import shortuuid


# 对原始密码进行加盐加密处理
def getmd5(telphone, rawpassword):
    obj = hashlib.md5(telphone.encode(encoding='UTF-8'))
    obj.update(rawpassword.encode(encoding='UTF-8'))
    result = obj.hexdigest()
    return result


# 允许上传的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


# 检查上传文件的格式是否符合规格
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream)
    return img_stream


def get6code():
    code = ''
    for i in range(0, 6):
        code += str(random.randint(1, 9))
    return code


# 加盐加密处理
if __name__ == '__main__':
    print(getmd5('666','777'))