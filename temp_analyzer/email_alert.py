"""
Usage:
  email_alert.py [options]

Options:
  -a,--all      send to all
  -f, --force    always send
  -h, --help     Show this screen.

"""

import smtplib
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
from docopt import docopt

from temp_analyzer.temp_plotter import get_temp

threshold_max = 26
threshold_min = 22.5

me = email = '6thFloorTemperature@gmx.com'
password = '6thFloorTemp'

POP3 = 'pop.gmx.com'
SMTP = 'mail.gmx.com'


def send_mail(args, alert):
    destination = [
        'nicolas.dickreuter@barclays.com',
    ]

    if args['--all']:
        destination.extend(
            [
                'claudio.nucera@barclays.com',
                'alan.j.james@barclays.com',
                'dimitris.kehagias@barclays.com',
                'dionisis.gonos@barclays.com',
                'Nora.bellavics@barclays.com',
                'Lynda.cairns@barclays.com',
                'richard.jefferies@barclays.com',
                'jeff.x.wood@barclays.com',
                'sergey.kurshev@barclays.com',
                'fotios.amaxopoulos@barclays.com'
            ]
        )

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    if alert:
        msg['Subject'] = 'Temperature ALERT 6th floor - action required'
    else:
        msg['Subject'] = 'Temperature report 6th floor'
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = email
    msg['To'] = ','.join(destination)
    msg.preamble = 'Temperature report'

    # We reference the image in the IMG SRC attribute by the ID we give it below
    msgText = MIMEText('Please find attached the latest temperature report..'
                       'Please open the attached file.<br><br>'
                       'Today max: {} C'
                       '<br>Today min: {} C'.format("%.2f" % max_val, "%.2f" % min_val),
                       'html')
    msg.attach(msgText)

    # This example assumes the image is in the current directory
    fp = open('chart.jpg', 'rb')
    msgImage = MIMEImage(fp.read())
    fp.close()

    # Define the image's ID as referenced above

    msg.attach(msgImage)

    # Send the email via our own SMTP server.
    s = smtplib.SMTP(SMTP)
    s.starttls()
    s.login(email, password)

    s.sendmail(me, destination, msg.as_string())
    s.quit()


if __name__ == '__main__':
    args = docopt(__doc__)

    df = get_temp()
    max_val = np.nanmax(df[['today sensor 1', 'today sensor 2']].values)
    min_val = np.nanmin(df[['today sensor 1', 'today sensor 2']].values)
    last_vals = df[['today sensor 1', 'today sensor 2']].dropna()[-1:].values

    alert = np.nanmax(last_vals) >= threshold_max or np.nanmin(last_vals) <= threshold_min

    if alert or args['--force']:
        print('Sending...')
        send_mail(args, alert)

    else:
        print('Not above threshold.')
