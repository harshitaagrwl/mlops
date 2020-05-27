import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
mail_content = "Your program Accuracy:"
#The mail addresses and password
sender_address = 'harshitaagarwal330@gmail.com'
sender_pass = input('enter password ')
receiver_address = 'dwiyash23@hotmail.com'
#Setup the MIME
message = MIMEMultipart()
message['From'] = 'harshitaagarwal330@gmail.com'
message['To'] = 'dwiyash23@hotmail.com'
message['Subject'] = 'Accuracy of model'
#The subject line
#The body and the attachments for the mail
message.attach(MIMEText(mail_content, 'plain'))
attach_file_name = 'accuracy.txt'
attach_file = open(attach_file_name, 'rb') # Open the file as binary mode
payload = MIMEBase('application', 'octate-stream')
payload.set_payload((attach_file).read())
encoders.encode_base64(payload) #encode the attachment
#add payload header with filename
payload.add_header('Content-Decomposition', 'attachment', filename=attach_file_name)
message.attach(payload)
#Create SMTP session for sending the mail
session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
session.starttls() #enable security
session.login(sender_address, sender_pass) #login with mail_id and password
text = message.as_string()
session.sendmail(sender_address, receiver_address, text)
session.quit()
print('Mail Sent')
