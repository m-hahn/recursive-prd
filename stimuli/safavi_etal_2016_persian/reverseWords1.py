with open("items_expt1.txt", "r") as inFile:
   for linenum, line in enumerate(inFile):
      line = line.strip()
      if len(line) < 3:
         print(line)
      else:
         line = line[1:-1].split(" ")[::-1]
         if True or linenum%5 in [2,4]:
            print((" ".join(line)).replace("_", " "))
