import pymysql
import uuid
import datetime
from iot.example.demo_sms_send import infosend

def selectbyplate(platenum):
    try:
        # 获取一个数据库连接，注意如果是UTF-8类型的，需要指定数据库
        conn = pymysql.connect(host='120.79.173.196', user='root', passwd='password', db='luffy', port=3306, charset='utf8')
        # 获取一个游标
        cur = conn.cursor()
        sql = "SELECT phone FROM car INNER JOIN `user` ON car.user_id = `user`.uid AND car.platenumber=%s"
        # # 插入站点信息, 入站时间以及出战时间
        cur.execute(sql, platenum)
        data = cur.fetchone()
        phone = str(data[0])
        # timenow = datetime.datetime.now()
        print(infosend(phone, platenum))
        # data = cur.fetchall()
        if data:
            sql_insert ="INSERT INTO records(records_id,platenumber, path_in_id,path_in_time) VALUES ('%s','%s',%d,NOW())"
            cur.execute(sql_insert % (uuid.uuid1(),platenum,1))
        # 执行sql语句
        conn.commit()
        cur.close()#关闭游标
        conn.close()#释放数据库资源
        return 1
    except  Exception :
        conn.rollback()
        return "查询失败"

if __name__ == '__main__':
    print(selectbyplate('云A7B5M6'))