
import difflib

with open("items_corrected.txt", "r") as inFile:
  corrected = enumerate(inFile.read().strip().split("\n"))
with open("items_original.txt", "r") as inFile:
  original = enumerate(inFile.read().strip().split("\n"))

with open("items_corrected_BD.txt", "w") as outFile:
   lineC, lineO = None, None
   while True:
      while not lineC:
        _, lineC = next(corrected)
      while not lineO:
        _, lineO = next(original)
      print(lineC, file=outFile)
      print("==========")
      print(lineC)
      print(lineO)
      if lineC != lineO:
          print("DIFFERENT")
          s = difflib.SequenceMatcher(None, lineC, lineO)
          for block in s.get_matching_blocks():
              print(lineC[block.a:block.a+block.size])
   
      _, lineC = next(corrected)
      _, lineO = next(original)
   
   
