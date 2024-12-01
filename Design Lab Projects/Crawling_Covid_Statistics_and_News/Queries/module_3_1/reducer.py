import sys

file = open(sys.argv[3], "w")
file.write(f"News From {sys.argv[1]} to {sys.argv[2]} :\n")
print(f"News From {sys.argv[1]} to {sys.argv[2]} :\n")
date_initial = None
sentence = ""

for line in sys.stdin:
    date, s = line.strip().split("\t")
    if date == date_initial:
        sentence = s + sentence
    else:
        if date_initial is not None:
            file.write(f"{date_initial} : {sentence}\n")
            print(f"{date_initial} : {sentence}\n")
        sentence = s
        date_initial = date

# Handle the last date outside the loop
if date_initial is not None:
    file.write(f"{date_initial} : {sentence}\n")
    print(f"{date_initial} : {sentence}\n")

file.close()
