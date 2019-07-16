from paths import WIKIPEDIA_HOME
import random
 

def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "hindi"
 
  stimuli = [] # each stimulus is a list of (line number - word) pair triples
  with open("stimuli/husain_etal_2014_hindi/final_items_all_tokenized.txt", "r") as inFile:
     stimulus = None
     next(inFile)
     linenum = 0
     for  line in inFile:
        line = line.strip().split("\t")
        if line[0] != stimulus:
          stimulus = line[0]
          stimuli.append([])
        stimuli[-1].append((linenum, line[4]))
        linenum += 1
  chunk = []
  chunk_line_numbers = []
  for i in range(1): # multiple runs mess up the recording system
     random.shuffle(stimuli)
     for s in stimuli:
         for w in s:
            chunk.append(w[1])
            chunk_line_numbers.append(w[0])

 # print(chunk_line_numbers)
#  quit()
  yield chunk, chunk_line_numbers

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)



if __name__ == '__main__':
    stream = test("hindi", 1)
    next(stream)

