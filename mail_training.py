import smtplib as s
p=s.SMTP('smtp.gmail.com',587)
print("Lets start e-mailing")
p.starttls()
email="harshitaagarwal330@gmail.com"
pwd=input('enter password')
p.login(email,pwd)
print("Login Sucessfull")
rec='dwiyash23@hotmail.com'
msg="hey!!\nYour model trained Successfully"  
p.sendmail(email,rec,msg)
p.close()
