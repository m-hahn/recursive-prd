from paths import WIKIPEDIA_HOME
import random
 

itos_labels = []
stoi_labels = {}

def load(language, section=None, partition="train", removeMarkup=True, tokenize=True, forGeneration=False):
  assert language == "russian"
  with open("../stimuli/Levy_etal_2013/expt"+section+"-tokenized.tsv", "r") as inFile:
     text = [x.split("\t") for x in inFile.read().strip().split("\n")]
  header = text[0]
  header = dict(zip(header, range(len(header))))
  text = text[1:]
  for line in text:
     #print(line)
     assert len(line) == len(header)
  chunk = []
  chunk_line_numbers = []
  if forGeneration:
      regions = []
  for linenum, line in enumerate(text):
     word = line[header['Word']].split("_")
     if forGeneration:
        region = line[header["Condition"]]+"_"+line[header["Region"]]

     if True:

        if tokenize:
           for word_ in word:
               chunk.append(word_.lower())
               chunk_line_numbers.append(linenum)
               if forGeneration:
                   regions.append(region)
        else:
           for char in " "+(" ".join(word)):
               chunk.append(char.lower())
               chunk_line_numbers.append(linenum)
               if forGeneration:
                  regions.append(region)
  print(chunk)
#  print(chunk)
  yield (chunk, chunk_line_numbers) if not forGeneration else (chunk, chunk_line_numbers, regions)

def test(language, removeMarkup=True, tokenize=True):
  return load(language, removeMarkup=removeMarkup, tokenize=tokenize)



if __name__ == '__main__':
    stream = test("english", "expt1")
    next(stream) 
