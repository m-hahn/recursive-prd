with open("items_expt1.txt", "r") as inFile:
   for line in inFile:
      line = line.strip()
      if len(line) < 3:
         print(line)
      else:
         line = line[1:-1].split(" ")[::-1]
         print((" ".join(line)).replace("_", " "))
