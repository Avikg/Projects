from datetime import datetime, timedelta
import sys

date1 = str(sys.argv[1])
date2 = str(sys.argv[2])
date1 = datetime.strptime(date1, "%Y-%m-%d")
date2 = datetime.strptime(date2, "%Y-%m-%d")

for line in sys.stdin:
    Date, sentence = line.strip().split("\t")
    try:
        Date = datetime.strptime(Date, "%Y-%m-%d")
    except ValueError:
        continue  

    if date1 <= Date <= date2:
        print(f"{Date.strftime('%Y-%m-%d')}\t{sentence}")
