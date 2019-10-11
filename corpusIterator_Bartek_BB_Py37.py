from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, partition="train", removeMarkup=True, tokenize=True, forGeneration=False):
  assert language == "english"
  with open("../stimuli/BarteketalJEP2011data/bb-spr06-data.txt", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  #print(text)
  header = ["subj","expt","item","condition","roi","word","correct","RT"] # bb-spr-dataprep.Rnw
  header = dict(zip(header, range(len(header))))
  for line in text:
     #print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  if forGeneration:
      regions = []
  for linenum, line in enumerate(text):
     word = line[header['word']]
     if line[header["correct"]] != "-":
#        print("SPECIAL", line)
        continue
     if forGeneration:
        region = line[header["condition"]]+"_"+line[header["roi"]]
#     words = words[1:-1].split("_")
     if True:
#        if word in ["N", "Y"]:
 #          continue

        if tokenize:
            if word[-1] in [".", ",", "?"]:
               chunk.append(word[:-1].lower())
               chunk.append(word[-1].lower())
               chunk_line_numbers.append(linenum)
               chunk_line_numbers.append(linenum)
               if forGeneration:
                   regions.append(region)
                   regions.append(region)
            else:
               chunk.append(word.lower())
               chunk_line_numbers.append(linenum)
               if forGeneration:
                   regions.append(region)
        else:
           for char in " "+word:
               chunk.append(char.lower())
               chunk_line_numbers.append(linenum)
               if forGeneration:
                  regions.append(region)

#  print(chunk)
  yield (chunk, chunk_line_numbers) if not forGeneration else (chunk, chunk_line_numbers, regions)

def test(language, removeMarkup=True, tokenize=True):
  return load(language, "test", removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("english")
    next(stream) 
