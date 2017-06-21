# -*- coding: utf-8 -*-
import time
import locale
from datetime import datetime
print(time.strftime("%a, %d %b %Y %H:%M:%S"))
locale.setlocale(locale.LC_TIME, "ru_RU")
print(time.strftime("%d %b %Y"))
# datetime_object = datetime.strptime('Jun 1 2005  1:33PM', '%b %d %Y %I:%M%p')
# datetime_object = datetime.strptime('1 Jun 2005  1:33PM', '%d %b %Y %I:%M%p')
# datetime_object = datetime.strptime('20 апреля 2017  1:33PM', '%d %b %Y %I:%M%p')
# print(datetime_object)
