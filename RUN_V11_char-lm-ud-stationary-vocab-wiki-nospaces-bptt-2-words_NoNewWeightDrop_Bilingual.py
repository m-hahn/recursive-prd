print("Character aware!")

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language1", type=str, default="english")
parser.add_argument("--language2", type=str, default="german")
parser.add_argument("--chosen_language", type=str, default="english")

parser.add_argument("--load_from", type=str, default=None) # 8066636
parser.add_argument("--section", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([128]))
parser.add_argument("--word_embedding_size", type=int, default=random.choice([512]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([1024]))
parser.add_argument("--layer_num", type=int, default=random.choice([2]))
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.05]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.05]))
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([1.0]))
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([50]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([1.0]))
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
parser.add_argument("--char_dec_hidden_dim", type=int, default=128)

model = "REAL_REAL"

import math

args=parser.parse_args()

#if "MYID" in args.save_to:
#   args.save_to = args.save_to.replace("MYID", str(args.myID))

#assert "word" in args.save_to, args.save_to

print(args)



if args.section == "E1":
   import corpusIterator_V11_E1_EN as corpusIterator_V11
elif args.section == "E1a":
   import corpusIterator_V11_E1a_EN as corpusIterator_V11
elif args.section == "E1_ColorlessGreen":
   import corpusIterator_V11_E1_EN_ColorlessGreen as corpusIterator_V11
elif args.section == "E3":
   import corpusIterator_V11_E3_DE as corpusIterator_V11
elif args.section == "E5":
   import corpusIterator_V11_E5_EN as corpusIterator_V11
elif args.section == "E6":
   import corpusIterator_V11_E6_EN as corpusIterator_V11
elif args.section == "E1_EitherVerb":
   import corpusIterator_V11_E1_EN_EitherVerb as corpusIterator_V11
elif args.section == "E3_Adapted":
   import corpusIterator_V11_E3_DE_Adapted as corpusIterator_V11
elif args.section == "E3_ColorlessGreen":
   import corpusIterator_V11_E3_DE_ColorlessGreen as corpusIterator_V11
else:
    assert False, args.section


def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

################################

char_vocab_path_1 = "vocabularies/"+args.language1.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path_1, "r") as inFile:
     itos_1 = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi_1 = dict([(itos_1[i],i) for i in range(len(itos_1))])
itos_total_1 = ["SOS","EOS", "OOV"] + itos_1

with open("vocabularies/char-vocab-wiki-"+args.language1, "r") as inFile:
     itos_chars_1 = [x for x in inFile.read().strip().split("\n")]
stoi_chars_1 = dict([(itos_chars_1[i],i) for i in range(len(itos_chars_1))])


itos_chars_total_1 = ["SOS", "EOS", "OOV"] + itos_chars_1

################################

char_vocab_path_2 = "vocabularies/"+args.language2.lower()+"-wiki-word-vocab-50000.txt"

with open(char_vocab_path_2, "r") as inFile:
     itos_2 = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:50000]]
stoi_2 = dict([(itos_2[i],i) for i in range(len(itos_2))])
itos_total_2 = ["SOS","EOS", "OOV"] + itos_2

with open("vocabularies/char-vocab-wiki-"+args.language1, "r") as inFile:
     itos_chars_2 = [x for x in inFile.read().strip().split("\n")]
stoi_chars_2 = dict([(itos_chars_2[i],i) for i in range(len(itos_chars_2))])


itos_chars_total_2 = ["SOS", "EOS", "OOV"] + itos_chars_2


################################

vocab_size = 50000
outVocabSize = 2*(3+vocab_size)


itos_total = ["EOS", "OOV", "SOS"] + itos_1 + ["EOS", "OOV", "SOS"] + itos_2 
assert len(itos_total) == outVocabSize



import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop


rnn = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()

rnn_parameter_names = [name for name, _ in rnn.named_parameters()]
print(rnn_parameter_names)
#quit()


rnn_drop = rnn #WeightDrop(rnn, layer_names=[(name, args.weight_dropout_in) for name, _ in rnn.named_parameters() if name.startswith("weight_ih_")] + [ (name, args.weight_dropout_hidden) for name, _ in rnn.named_parameters() if name.startswith("weight_hh_")])

output_1 = torch.nn.Linear(args.hidden_dim,50003).cuda()
output_2 = torch.nn.Linear(args.hidden_dim,50003).cuda()


word_embeddings = torch.nn.Embedding(num_embeddings=3+len(itos_1)+3+len(itos_2), embedding_dim=args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)

train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')

modules = [rnn, output_1, output_2, word_embeddings]


character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total_1)+ len(itos_chars_total_2), embedding_dim=args.char_emb_dim).cuda()


char_composition = torch.nn.LSTM(args.char_emb_dim, args.char_enc_hidden_dim, 1, bidirectional=True).cuda()
char_composition_output = torch.nn.Linear(2*args.char_enc_hidden_dim, args.word_embedding_size).cuda()

#char_decoder_rnn = torch.nn.LSTM(args.char_emb_dim + args.hidden_dim, args.char_dec_hidden_dim, 1).cuda()
#char_decoder_output = torch.nn.Linear(args.char_dec_hidden_dim, len(itos_chars_total))


modules += [character_embeddings, char_composition, char_composition_output] #, char_decoder_rnn, char_decoder_output]
def parameters():
   for module in modules:
       for param in module.parameters():
            yield param

parameters_cached = [x for x in parameters()]


learning_rate = args.learning_rate

optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9

#named_modules = {"rnn" : rnn, "output" : output, "word_embeddings" : word_embeddings, "optim" : optim}

#if args.load_from is not None:
#  checkpoint = torch.load(MODELS_HOME+"/"+args.load_from+".pth.tar")
#  for name, module in named_modules.items():
 #     module.load_state_dict(checkpoint[name])




if args.load_from is not None:
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language1+"AND"+args.language2+"_"+__file__.replace("RUN_V11_","")+"_code_"+args.load_from+".txt")
  for i in range(len(checkpoint["components"])):
      modules[i].load_state_dict(checkpoint["components"][i])
else:
  assert False

from torch.autograd import Variable






def prepareDatasetChunks(data, batchSizeHere=args.batchSize, train=True, stoi=stoi_1 if args.chosen_language == args.language1 else stoi_2, stoi_chars=stoi_chars_1 if args.chosen_language == args.language1 else stoi_chars_2):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      numerified_chars = []
      line_numbers = []
      for chunk, chunk_line_numbers in data:
       for char, linenum in zip(chunk, chunk_line_numbers):
         count += 1
         numerified.append((stoi[char]+3 if char in stoi else 2))
         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])
         line_numbers.append(linenum)

       if len(numerified) > (args.batchSize*args.sequence_length):
         sequenceLengthHere = args.sequence_length

         cutoff = int(len(numerified)/(args.batchSize*sequenceLengthHere)) * (args.batchSize*sequenceLengthHere)
         numerifiedCurrent = numerified[:cutoff]
         numerifiedCurrent_chars = numerified_chars[:cutoff]

         for i in range(len(numerifiedCurrent_chars)):
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i][:15] + [1]
            numerifiedCurrent_chars[i] = numerifiedCurrent_chars[i] + ([0]*(16-len(numerifiedCurrent_chars[i])))


         numerified = numerified[cutoff:]
         numerified_chars = numerified_chars[cutoff:]
       
         line_numbersCurrent = line_numbers[:cutoff]
         line_numbers = line_numbers[cutoff:]
        
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(args.batchSize, -1, sequenceLengthHere, 16).transpose(0,1).transpose(1,2).cuda()

         line_numbersCurrent = torch.LongTensor(line_numbersCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], numerifiedCurrent_chars[i], line_numbersCurrent[i]
         hidden = None
       else:
         print("Skipping")


completeData = []



hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)
beginning = None

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())




def forward(numericAndLineNumbers, train=True, printHere=False):
      global hidden
      global beginning
      global beginning_chars
      if hidden is not None:
          hidden1 = Variable(hidden[0]).detach()
          hidden2 = Variable(hidden[1]).detach()
          forRestart = bernoulli.sample()
          hidden1 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden1)
          hidden2 = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroHidden, hidden2)
          hidden = (hidden1, hidden2)
          beginning = torch.where(forRestart.unsqueeze(0) == 1, zeroBeginning, beginning)
          beginning_chars = torch.where(forRestart.unsqueeze(0).unsqueeze(2) == 1, zeroBeginning_chars, beginning_chars)
      elif hidden is None:
          hidden = None
          beginning = zeroBeginning
          beginning_chars = zeroBeginning_chars

      numeric, numeric_chars, lineNumbers = numericAndLineNumbers


      numeric = torch.cat([beginning, numeric], dim=0)

      numeric_chars = torch.cat([beginning_chars, numeric_chars], dim=0)

      beginning = numeric[numeric.size()[0]-1].view(1, args.batchSize)
      beginning_chars = numeric_chars[numeric_chars.size()[0]-1].view(1, args.batchSize, 16)


      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_chars = Variable(numeric_chars[:-1], requires_grad=False)
      if args.chosen_language == args.language2:
         input_tensor_chars = input_tensor_chars + len(itos_chars_total_1)

      # TODO it is weird (BUG) that this is only done for characters, not for words

      embedded_chars = input_tensor_chars.transpose(0,2).transpose(2,1)
      embedded_chars = embedded_chars.contiguous().view(16, -1)
      _, embedded_chars = char_composition(character_embeddings(embedded_chars), None)
      embedded_chars = embedded_chars[0].view(2, args.sequence_length, args.batchSize, args.char_enc_hidden_dim)
      #print(embedded_chars.size())

      embedded_chars = char_composition_output(torch.cat([embedded_chars[0], embedded_chars[1]], dim=2))

      embedded = word_embeddings(input_tensor)
      embedded = torch.cat([embedded, embedded_chars], dim=2)

      out, hidden = rnn_drop(embedded, hidden)
#      if train:
#          out = dropout(out)


      logits = (output_1 if args.chosen_language == args.language1 else output_2)(out)




      log_probs = logsoftmax(logits)

      
      lossTensor = print_loss(log_probs.view(-1, 50003), target_tensor.view(-1)).view(-1, args.batchSize)
      losses = lossTensor.data.cpu().numpy()




      for i in range(0,args.sequence_length): #range(1,maxLength+1): # don't include i==0
         j = 0
         numericCPU = numeric.cpu().data.numpy()
         lineNum = int(lineNumbers[i][j])

         print (i, (itos_total_1 if args.chosen_language == args.language1 else itos_total_2)[numericCPU[i+1][j]], losses[i][j], lineNum)

         while lineNum >= len(completeData):
             completeData.append([[], 0])
         completeData[lineNum][0].append((itos_total_1 if args.chosen_language == args.language1 else itos_total_2)[numericCPU[i+1][j]])
         completeData[lineNum][1] += losses[i][j]


      return None, target_tensor.view(-1).size()[0]




import time

testLosses = []

if True:
   rnn_drop.train(False)


   test_data = corpusIterator_V11.load(args.chosen_language, tokenize=True)

   print("Got data")
   test_chars = prepareDatasetChunks(test_data, train=False)



     
   test_loss = 0
   test_char_count = 0
   counter = 0
   hidden, beginning = None, None
   while True:
       counter += 1
       try:
          numeric = next(test_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       test_char_count += numberOfCharacters
   testLosses.append(test_loss/test_char_count)
   print(testLosses)


with open("output/V11_"+args.section+"_"+args.chosen_language+"_"+args.load_from, "w") as outFile:
   print("\t".join(["LineNumber", "RegionLSTM", "Surprisal"]), file=outFile)
   for num, entry in enumerate(completeData):
     print("\t".join([str(x) for x in [num, "".join(entry[0]), entry[1]]]), file=outFile)

