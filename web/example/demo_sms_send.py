# -*- coding: utf-8 -*-


from aliyunsdkdysmsapi.request.v20170525 import SendSmsRequest
from aliyunsdkcore.client import AcsClient
import uuid
from aliyunsdkcore.profile import region_provider
import example.const as const
import json

"""
短信业务调用接口示例，版本号：v20170525

Created on 2017-06-12

"""
# try:
#     reload(sys)
#     s
# except NameError:
#     pass
# except Exception as err:
#     raise err

# 注意：不要更改
REGION = "cn-hangzhou"
PRODUCT_NAME = "Dysmsapi"
DOMAIN = "dysmsapi.aliyuncs.com"


acs_client = AcsClient(const.ACCESS_KEY_ID, const.ACCESS_KEY_SECRET, REGION)
region_provider.add_endpoint(PRODUCT_NAME, REGION, DOMAIN)


# 短信发送函数, param用于指定验证码之类的东西
def send_sms(business_id, phone_numbers, sign_name, template_code, template_param=None):
    smsRequest = SendSmsRequest.SendSmsRequest()
    # 申请的短信模板编码,必填
    smsRequest.set_TemplateCode(template_code)
    # 短信模板变量参数
    if template_param is not None:
        smsRequest.set_TemplateParam(template_param)
    # 设置业务请求流水号，必填。
    smsRequest.set_OutId(business_id)
    # 短信签名
    smsRequest.set_SignName(sign_name)
    # 数据提交方式
    # smsRequest.set_method(MT.POST)
    # 数据提交格式
    # smsRequest.set_accept_format(FT.JSON)
    # 短信发送的号码列表，必填。
    smsRequest.set_PhoneNumbers(phone_numbers)
    # 调用短信发送接口，返回json
    smsResponse = acs_client.do_action_with_exception(smsRequest)
    # TODO 业务处理
    return smsResponse


# 发送验证码 返回ok表示验证码发送成功
def codesend(phone, code):
    __business_id = uuid.uuid1()
    params = "{\"code\":"+code+",\"product\":\"云通信\"}"
    message = send_sms(__business_id, phone, "宋晨明", "SMS_139240772", params)
    message = eval(str(message, encoding='utf-8'))
    # {'Message': 'OK', 'RequestId': '4914D11D-7985-4896-BECF-C2FE98F18304', 'BizId': '321604643730506774^0','Code': 'OK'}
    result = message.get("Code")
    return result


# 测试短信发送案例
if __name__ == '__main__':
    # __business_id = uuid.uuid1()
    # params = "{\"code\":\"666666\",\"product\":\"云通信\"}"
    # message = send_sms(__business_id, "17803415224", "宋晨明", "SMS_139240772", params)
    # message = eval(str(message, encoding='utf-8'))
    # # {'Message': 'OK', 'RequestId': '4914D11D-7985-4896-BECF-C2FE98F18304', 'BizId': '321604643730506774^0','Code': 'OK'}
    # result = message.get("Code")
    print(codesend('17803415224', '555555'))

   
    
    

