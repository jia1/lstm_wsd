import time
import re
import sys

import smtplib
from email.mime.text import MIMEText
import socket

host_name = socket.gethostname()

if len(sys.argv) < 3:
    sys.exit('run_id password')

pwd = sys.argv[2]
run_id = sys.argv[1] 

path = '/home/salomons/project/wsd/smac-output/scenario/live-rundata-' + str(run_id) + '.json'

best = 1000.0


def send_email(best):
    print 'New best found: %f. Sending email.' % best
    msg = MIMEText('computer name: %s\nrun_id: %d' % (host_name, run_id))

# me == the sender's email address
# you == the recipient's email address
    msg['Subject'] = 'New best: %f' % best
    msg['From'] = from_ = 'salomons@net.chalmers.se'
    msg['To'] = to_ = 'hans.salomonsson@outlook.com'

# Send the message via our own SMTP server, but don't include the
# envelope header.
    s = smtplib.SMTP('smtp.office365.com', 587)
    s.starttls()
    s.login("salomons@net.chalmers.se", pwd)
    s.sendmail(from_, [to_], msg.as_string())
    s.quit()

while True:
    with open(path, 'r') as file:
        doc = file.read()

    qualities = re.findall('\"r-quality\" : [+-d]+', doc)
    q = []
    for quality in qualities:
        q.append(float(quality.split(' ')[-1].replace(',', '')))

    if q:
        candidate = min(q)
        if candidate < best:
            best = candidate
            send_email(best)

    time.sleep(240)
