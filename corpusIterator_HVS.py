from paths import WIKIPEDIA_HOME
import random
 

def load(language, section, partition="train", removeMarkup=True, tokenize=True):
  assert language == "hindi"
 
  stimuli = [] # each stimulus is a list of item - integer - word - region pair triples
  with open("stimuli/husain_etal_2014_hindi/final_items_all.txt", "r") as inFile:
     typ = None
     for line in inFile:
        if line.startswith("#"):
            typ = line.strip()
        elif line.startswith("?"):
            continue
        elif len(line) > 2:
           stimuli.append([])
           text = line.strip()
           text = [x.split("@") for x in text.split(" ")]
           words = [x[0].replace(",", "_,").split("_") for x in text]
           regions = [x[1] if len(x) > 1 else "NONE" for x in text]
           words = zip(words, regions)
           i = 0
           for x, r in words:
              if r != "NONE":
#                 if ("RC1" in typ and r == "RCVerb") or ("RC2" in typ and r in ["NP1", "RCPn"]) or ("PP1" in typ and r == "HeadNP") or("CP1" in typ and r == "CPLightVerb"):
                    
                 for y in x:
                #   if y not in stoi:
#                      print(typ, y, y in stoi, r) #([[(y,y in stoi) for y in x] for x in words])
                      print("\t".join((typ, str(i), y, r)))
                      stimuli[-1].append((typ, i, y, r))
                      i += 1
  quit()  


  chunk = []

  chunk_line_numbers = []
  for linenum, line in enumerate(text):
     words = line[header['word']]
     words = words.split("_")
     for word in words:

        if tokenize:
            if word[-1] in [".", ","]:
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

  print(chunk)
  yield chunk, chunk_line_numbers

def test(language, section=1, removeMarkup=True):
  return load(language, section, "test", removeMarkup=removeMarkup)



if __name__ == '__main__':
    stream = test("hindi", 1)
    next(stream)

