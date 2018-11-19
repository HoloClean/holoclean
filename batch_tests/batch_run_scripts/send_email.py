
import smtplib
import sys

password = 'theowantsitsoon!' 
msg = sys.argv[1]

smtpObj = smtplib.SMTP('smtp.gmail.com',587)
smtpObj.ehlo()
smtpObj.starttls()
smtpObj.login('rekords.experiments@gmail.com',password)
smtpObj.sendmail('rekords.experiments@gmail.com','zguo74@wisc.edu',"Subject: Holoclean1 [{}] is done!".format(msg))
smtpObj.quit()
