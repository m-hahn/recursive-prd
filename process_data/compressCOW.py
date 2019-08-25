import sys

path = sys.argv[1].replace(".xml.gz", "")

import html
import gzip
counter = 0
buff = []
with gzip.open(path+'.xml.gz','r') as f:
 with open(path+'.txt','w') as outFile:
  for line in f:
    counter += 1
    if counter % 500000 == 0:
       print(counter)

    try:    
        line = line.decode().strip()
    except UnicodeDecodeError:
        print("Unicode Error")
        print(line)
        line = line.decode("iso-8859-1").strip()
        print(line)
    if line == "</s>":
       print((" ".join(buff)).lower(), file=outFile)
       buff = []
    if not line.startswith("<"):
       buff.append(line[:line.index("\t")])
       if buff[-1].startswith("&"):
           buff[-1] = html.unescape(buff[-1])
#    else:
 #      print('got line', line, line == "</s>")

