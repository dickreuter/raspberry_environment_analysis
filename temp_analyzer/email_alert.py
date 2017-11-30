
me = email = '6thFloorTemperature@gmx.com'
password = '6thFloorTemp'

POP3 = 'pop.gmx.com'
SMTP = 'mail.gmx.com'
destination = 'Nicolas.Dickreuter@barcap.com, Claudio.Nucera@barcap.com'

import smtplib

# Here are the email package modules we'll need
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

COMMASPACE = ', '

# Create the container (outer) email message.
msg = MIMEMultipart()
msg['Subject'] = 'Temperature report'
# me == the sender's email address
# family = the list of all recipients' email addresses
msg['From'] = email
msg['To'] = COMMASPACE.join(destination)
msg.preamble = 'Temperature report'


file = 'temp_charts.jpg'
fp = open(file, 'rb')
img = MIMEImage(fp.read())
fp.close()
msg.attach(img)

# Send the email via our own SMTP server.
s = smtplib.SMTP(SMTP)
s.sendmail(me, destination, msg.as_string())
s.quit()