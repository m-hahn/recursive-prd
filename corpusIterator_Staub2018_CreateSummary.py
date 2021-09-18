from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

idToSentence = {}
idToSubject = {}

with open("/u/scr/mhahn/STIMULI/Staub2018.tsv", "w") as outFile:
  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
  if True:
     with open("../stimuli/Staub_2018/items.txt", "r") as inFile:
        text = [x for x in inFile.read().strip().split("\n")]
     stimuli = []
   
     lastLabel = None
     for line in text:
        line = line.strip()
        print(line)
        itemID = line[:line.index(" ")]
        line = line[line.index(" ")+1:]
        if itemID.startswith("*"):
           assert line.endswith(")")
           alternativeNoun = line[line.rfind("(")+1:-1]
           line = line[:line.rfind("(")].strip()
        else:
           alternativeNoun = None
        itemID = itemID.strip(".").strip("*")
        part1 = line[:line.index("(")].strip().split(" ")
        part2 = line[line.index("(")+1:line.index(")")].strip().split(" ")
        part3 = line[line.index(")")+1:].strip().split(" ")
        part1 = [[x, i, f"Pre_{i+1}"] for i,x in enumerate(part1)]
        part2 = [[x, i, f"Object_{i+1}"] for i,x in enumerate(part2)]
        part3 = [[x, i, f"Post_{i+1}"] for i,x in enumerate(part3)]
        part3[0][2] = "Verb_1"
        print(itemID, part1, part2, part3, alternativeNoun)
        # Experiment 1, RC version
        # Experiment 1, CC version
 #  print("\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)
       
        quit()
        itemID = line[0].strip().strip(".")
        print(line)
        sentence = line[1]
        sentence = sentence.lower() + " "
        firstSlash = sentence.index("/")
        secondSlash = sentence.find("/", firstSlash+1)
        firstStart = sentence.rfind(" ", 0, firstSlash)
        firstEnd = sentence.find(" ", firstSlash)
   
        secondStart = sentence.rfind(" ", 0, secondSlash)
        secondEnd = sentence.find(" ", secondSlash)
   
        adverb = sentence.find("ly", secondEnd)
        verbStart = adverb+3
        verbEnd = sentence.find(" ", verbStart)
        verb = sentence[verbStart:verbEnd]
        assert sentence.count(verb) == 1
   
   
        part1 = sentence[:firstStart]
        part2 = sentence[firstEnd:secondStart]
        part3 = sentence[secondEnd:]
        w11 = sentence[firstStart:firstSlash].strip()
        w12 = " "+sentence[firstSlash+1:firstEnd]
   
        w21 = sentence[secondStart:secondSlash].strip()
        w22 = " "+sentence[secondSlash+1:secondEnd]
  
        part3_a, part3_b = part3[:part3.index("/")].strip(), part3[part3.index("/"):].strip()
        part3_ba, part3_bb = part3_b[:part3_b.index(" ")].strip().strip("/"), part3_b[part3_b.index(" "):].strip().replace(".", " .")
        part1 = part1.replace(".", " .").split(" ")
        for i in range(len(part1)):
           if part1[i] == ".":
              part1[i+1] = part1[i+1][0].upper() + part1[i+1][1:]
              part1 = part1[i+1:]
              break
        part2 = part2.strip().split(" ")
        part3_a = part3_a.strip().split(" ")
        part3_bb = part3_bb.strip().split(" ")
        print([part1, w11, part2, w21, part3_a, part3_ba, part3_bb])
        part1 = [x for x in [y.strip() for y in part1] if len(x) > 0]
        part2 = [x for x in [y.strip() for y in part2] if len(x) > 0]
        part3_a = [x for x in [y.strip() for y in part3_a] if len(x) > 0]
        part3_bb = [x for x in [y.strip() for y in part3_bb] if len(x) > 0]
        w11 = w11.strip()
        w12 = w12.strip()
        w21 = w11.strip()
        w22 = w22.strip()
        for condition in ["11","12","21","22"]:
           sentID = itemID+"-"+condition
           condition_name = "-".join([["plau_sent", "impl_sent"][int(condition[0])-1], ["plau_dist", "impl_dist"][int(condition[1])-1]])
           counter = 0
           for i, word in enumerate(part1):
             counter+=1
             region = f"Pre_{i}"
             print("\t".join([str(q) for q in [sentID, itemID, condition_name, region, word, counter]]), file=outFile)
           counter+=1
           word = (w11 if condition[0] == "1" else w12)
           region = "noun"
           print("\t".join([str(q) for q in [sentID, itemID, condition_name, region, word, counter]]), file=outFile)
           for i, word in enumerate(part2):
             counter+=1
             region = f"Mid_{i}"
             print("\t".join([str(q) for q in [sentID, itemID, condition_name, region, word, counter]]), file=outFile)
           counter+=1
           word = (w21 if condition[1] == "1" else w22)
           region = "distractor"
           print("\t".join([str(q) for q in [sentID, itemID, condition_name, region, word, counter]]), file=outFile)
           for i, word in enumerate(part3_a):
             counter+=1
             region = f"Adverb_{i}"
             print("\t".join([str(q) for q in [sentID, itemID, condition_name, region, word, counter]]), file=outFile)
           counter+=1
           word = part3_ba
           region = "critical"
           print("\t".join([str(q) for q in [sentID, itemID, condition_name, region, word, counter]]), file=outFile)
           for i, word in enumerate(part3_bb):
             counter+=1
             region = f"Post_{i}"
             print("\t".join([str(q) for q in [sentID, itemID, condition_name, region, word, counter]]), file=outFile)
#  print(sentID, "\t".join(["Sentence", "Item", "Condition", "Region", "Word", "NumInSent"]), file=outFile)

      
