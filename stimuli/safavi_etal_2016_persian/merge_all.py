with open("items_corrected_BD.txt", "r") as c:
   corr = c.read().strip().split("\n")
with open("items_original_all.txt", "r") as o:
   orig = o.read().strip().split("\n")

with open("items_merged_beforeEdit.txt", "w") as o:
   for part in range(2):
      print("Part "+str(part+1), file=o)
      for i in range(36):
        assert str(i+1) == orig[181*part+i*5+1], (i, orig[181*part+i*5+1])
        assert str(i+1) == corr[109*part+i*3+1], (i, corr[109*part+i*3+1])
        print(i+1, file=o)
        print(orig[181*part+1+i*5+1], file=o)
        print(corr[109*part+1+i*3+1], file=o)
        print(orig[181*part+1+i*5+3], file=o)
        print(corr[109*part+1+i*3+2], file=o)
   
