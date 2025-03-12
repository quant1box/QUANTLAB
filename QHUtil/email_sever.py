import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(subject, body, to_email, smtp_server, smtp_port, sender_email, sender_password):
    # 创建邮件对象
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # 添加邮件正文
    msg.attach(MIMEText(body, 'plain'))

    # 连接到SMTP服务器
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        # 登录邮箱
        server.starttls()
        server.login(sender_email, sender_password)

        # 发送邮件
        server.sendmail(sender_email, to_email, msg.as_string())

# # 设置邮件参数
# subject = 'Test Email'
# body = 'This is a test email sent from Python.'
# to_email = 'recipient@example.com'
# smtp_server = 'smtp.example.com'
# smtp_port = 587
# sender_email = 'your_email@example.com'
# sender_password = 'your_email_password'

# # 发送邮件
# send_email(subject, body, to_email, smtp_server, smtp_port, sender_email, sender_password)
