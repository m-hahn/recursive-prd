s1, s2 = None, None
with open("items_expt2_edited.txt", "r") as inFile:
   with open("items_expt2_edited_reverse_onlyBD.txt", "w") as outFile:
     for linenum, line in enumerate(inFile):
        if linenum % 5 in [1, 3]:
           continue
        line = line.strip()
        if len(line) > 4:
           if s1 == None:
             s1 = line[0]
             s2 = line[-1]
           line = line.replace(s1, " ").replace(s2, " ")
           line = " ".join([x for x in line.strip().split(" ")[::-1] if len(x) > 0])
        print(line, file=outFile)
