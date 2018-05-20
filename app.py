"""
Usage:
  app.py email LIMIT [options]
  app.py predict
  app.py train
  app.py email_predict LIMIT [options]

Options:
  -a,--all       send to all
  -f, --force    always send
  -n, --any      send if any temperature today surprassed the max not just the last
  -h, --help     Show this screen.

"""
import json
import os
import smtplib
import sys
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import numpy as np
from docopt import docopt

from prediction_models.rnn_nicolas import NeuralNetworkNicolas, predict
from temp_analyzer.temp_plotter import get_temp

me = email = '6thFloorTemperature@gmx.com'
password = '6thFloorTemp'

POP3 = 'pop.gmx.com'
SMTP = 'mail.gmx.com'


def send_mail(args, alert):
    destination = [
        'nicolas.dickreuter@barclays.com',
    ]

    if args['--all']:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'contacts.json')) as json_data:
            d = json.load(json_data)
            destination.extend(d)

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    if alert==True:
        msg['Subject'] = 'Temperature ALERT 6th floor - action required'
    elif alert==False:
        msg['Subject'] = 'Temperature report 6th floor'
    elif alert=='prediction':
        msg['Subject'] = 'Temperature report 6th floor - neural network prediction alert'
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = email
    msg['To'] = ','.join(destination)
    msg.preamble = 'Temperature report'

    # We reference the image in the IMG SRC attribute by the ID we give it below
    msgText = MIMEText('Please find attached the latest temperature report.'
                       '<br><br>'
                       'Today max: {} C'
                       '<br>Today min: {} C'
                       '<BR> <BR> To contribute please fork: https://github.com/dickreuter/raspberry_environment_analysis'
                       .format("%.2f" % max_val, "%.2f" % min_val), 'html')
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

    if args['train']:
        p = NeuralNetworkNicolas()
        p.train()
        sys.exit()

    if args['predict']:
        prediction = predict()
        print(prediction)
        sys.exit()

    if args['email']:
        threshold_max = float(args['LIMIT'])
        threshold_min = 20.5

        df = get_temp(threshold_min, threshold_max)
        max_val = np.nanmax(df[['today sensor 1', 'today sensor 2']].values)
        min_val = np.nanmin(df[['today sensor 1', 'today sensor 2']].values)
        last_vals = df[['today sensor 1', 'today sensor 2']].dropna()[-1:].values

        alert = np.nanmax(last_vals) >= threshold_max or np.nanmin(last_vals) <= threshold_min

        any_alert = False
        if args['--any']:
            any_alert = max_val >= threshold_max or min_val <= threshold_min

        if alert or args['--force'] or (args['--any'] and any_alert):
            print('Sending...')
            send_mail(args, alert)

        else:
            print('Not above threshold.')
