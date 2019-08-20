import html
import gzip
counter = 0
buff = []
with gzip.open('/u/scr/mhahn/CORPORA/COW/decow16bx/decow16bx01.xml.gz','r') as f:
 with open('/u/scr/mhahn/CORPORA/COW/decow16bx/decow16bx01.txt','w') as outFile:
  for line in f:
    counter += 1
    if counter % 10000 == 0:
       print(counter)
    
    line = line.decode().strip()
    if line == "</s>":
       print((" ".join(buff)).lower(), file=outFile)
       buff = []
    if not line.startswith("<"):
       buff.append(line[:line.index("\t")])
       if buff[-1].startswith("&"):
           buff[-1] = html.unescape(buff[-1])
#    else:
 #      print('got line', line, line == "</s>")

