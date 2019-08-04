from paths import WIKIPEDIA_HOME
import random
 
import codecs
def load(language, partition="train", removeMarkup=True, tokenize=True):
  assert language == "german"
  with open("/u/scr/mhahn/CODE/stimuli/V11/colorlessGreen_German.txt", "r") as inFile:
     text = [x.strip().split("\t") for x in inFile.read().strip().split("\n")]
  header = text[0]
  text = text[1:]
  header = dict(zip(header, range(len(header))))
  chunk = []

  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     print(line)
     assert len(line) == len(header)
     words = line[header['Word']]
     words = words.split("_")
     for word in words:
        if len(word) == 0:
           continue
        if tokenize:
            if word[-1] in [".", ","] and len(word) > 1:
               chunk.append(word[:-1].lower())
               chunk.append(word[-1].lower())
               chunk_line_numbers.append(linenum)
               chunk_line_numbers.append(linenum)
            else:
               chunk.append(word.lower())
               chunk_line_numbers.append(linenum)
        else:
           for char in " "+word:
               chunk.append(char.lower())
               chunk_line_numbers.append(linenum)
  yield chunk, chunk_line_numbers

def test(language, removeMarkup=True):
  return load(language, "test", removeMarkup=removeMarkup)



if __name__ == '__main__':
    stream = test("german")
    next(stream)

