print("Character aware!")


# Derived from autoencoder.py, uses noise

# Character-aware version of the `Tabula Rasa' language model
# char-lm-ud-stationary-vocab-wiki-nospaces-bptt-2-words_NoNewWeightDrop.py
# Adopted for English and German
import sys

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--language", dest="language", type=str, default="english")
parser.add_argument("--load-from", dest="load_from", type=str)
#parser.add_argument("--save-to", dest="save_to", type=str)

import random

parser.add_argument("--batchSize", type=int, default=random.choice([32])) # , 256, 128
parser.add_argument("--word_embedding_size", type=int, default=random.choice([256]))
parser.add_argument("--hidden_dim", type=int, default=random.choice([512])) # 256, 512, 
parser.add_argument("--layer_num", type=int, default=random.choice([1])) # 1, 
parser.add_argument("--weight_dropout_in", type=float, default=random.choice([0.2]))
parser.add_argument("--weight_dropout_out", type=float, default=random.choice([0.01])) # , 0.2, 0.3
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.2])) # , 0.1, 0.2, 0.3
#parser.add_argument("--char_noise_prob", type = float, default=random.choice([0.0]))
parser.add_argument("--learning_rate", type = float, default= random.choice([0.1])) # 0.1, 0.15, 0.2, 0.3, 0.5, 2.0
parser.add_argument("--myID", type=int, default=random.randint(0,1000000000))
parser.add_argument("--sequence_length", type=int, default=random.choice([30]))
parser.add_argument("--verbose", type=bool, default=False)
parser.add_argument("--lr_decay", type=float, default=random.choice([ 1.0])) # 0.95, 0.98, 0.99
parser.add_argument("--lr_decay_after_failure", type=float, default=random.choice([0.7])) # , 1.0, 0.9, 0.95
parser.add_argument("--char_emb_dim", type=int, default=128)
parser.add_argument("--char_enc_hidden_dim", type=int, default=64)
parser.add_argument("--char_dec_hidden_dim", type=int, default=128)
parser.add_argument("--celltype", type=str, default=random.choice(["gru"])) # , "lstm"
parser.add_argument("--infinite_context", type=bool, default=random.choice([True])) # , False

model = "REAL_REAL"

import math

args=parser.parse_args()


print(args)



import corpusIterator_PTB_Deps



def plus(it1, it2):
   for x in it1:
      yield x
   for x in it2:
      yield x

char_vocab_path = "vocabularies/ptb-word-vocab.txt"

with open(char_vocab_path, "r") as inFile:
     itos = [x.split("\t")[0] for x in inFile.read().strip().split("\n")[:10000]]
stoi = dict([(itos[i],i) for i in range(len(itos))])


itos_total = ["<SOS>", "<EOS>", "OOV"] + itos
stoi_total = dict([(itos_total[i],i) for i in range(len(itos_total))])


with open("vocabularies/char-vocab-wiki-"+args.language, "r") as inFile:
     itos_chars = [x for x in inFile.read().strip().split("\n")]
stoi_chars = dict([(itos_chars[i],i) for i in range(len(itos_chars))])


itos_chars_total = ["<SOS>", "<EOS>", "OOV"] + itos_chars


import random


import torch

print(torch.__version__)

#from weight_drop import WeightDrop

if args.celltype == "gru":
   rnn_decoder = torch.nn.GRU(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()
elif args.celltype == "lstm":
   rnn_decoder = torch.nn.LSTM(2*args.word_embedding_size, args.hidden_dim, args.layer_num).cuda()
else:
   assert False

output = torch.nn.Linear(args.hidden_dim, len(itos)+3).cuda()

word_embeddings = torch.nn.Embedding(num_embeddings=len(itos)+3, embedding_dim=2*args.word_embedding_size).cuda()

logsoftmax = torch.nn.LogSoftmax(dim=2)
softmax = torch.nn.Softmax(dim=2)

attention_softmax = torch.nn.Softmax(dim=0)


train_loss = torch.nn.NLLLoss(ignore_index=0)
print_loss = torch.nn.NLLLoss(size_average=False, reduce=False, ignore_index=0)
char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)


train_loss_chars = torch.nn.NLLLoss(ignore_index=0, reduction='sum')


attention_proj = torch.nn.Linear(2*args.hidden_dim + 2*args.word_embedding_size, args.hidden_dim).cuda()
#attention_layer = torch.nn.Bilinear(args.hidden_dim, args.hidden_dim, 1, bias=False).cuda()
attention_proj.weight.data.fill_(0)



attention_out = torch.nn.Linear(args.hidden_dim, 1, bias=False).cuda()
#attention_out.weight.data.fill_(0)


toKeys = torch.nn.Linear(args.hidden_dim, args.hidden_dim, bias=False).cuda()
toValues = torch.nn.Linear(args.hidden_dim, args.hidden_dim, bias=False).cuda()


modules = [rnn_decoder, output, word_embeddings, attention_proj, toKeys, toValues, attention_out]
tanh = torch.nn.Tanh()

#character_embeddings = torch.nn.Embedding(num_embeddings = len(itos_chars_total)+3, embedding_dim=args.char_emb_dim).cuda()
#
#char_composition = torch.nn.LSTM(args.char_emb_dim, args.char_enc_hidden_dim, 1, bidirectional=True).cuda()
#char_composition_output = torch.nn.Linear(2*args.char_enc_hidden_dim, args.word_embedding_size).cuda()
#
#char_decoder_rnn = torch.nn.LSTM(args.char_emb_dim + args.hidden_dim, args.char_dec_hidden_dim, 1).cuda()
#char_decoder_output = torch.nn.Linear(args.char_dec_hidden_dim, len(itos_chars_total))
#
#
#modules += [character_embeddings, char_composition, char_composition_output, char_decoder_rnn, char_decoder_output]
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
  checkpoint = torch.load("/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__.replace("_sample_debug", "")+"_code_"+str(args.load_from)+".txt")
  for i in range(len(checkpoint["components"])):
      modules[i].load_state_dict(checkpoint["components"][i])

from torch.autograd import Variable


# ([0] + [stoi[training_data[x]]+1 for x in range(b, b+sequence_length) if x < len(training_data)]) 

#from embed_regularize import embedded_dropout



def prepareDatasetChunks(data, train=True):
      numeric = [0]
      count = 0
      print("Prepare chunks")
      numerified = []
      numerified_chars = []
      global hidden
      for chunk in data:
       #print(len(chunk))
       for char in chunk:
#         if char == " ":
 #          continue
         count += 1
#         if count % 100000 == 0:
#             print(count/len(data))
         numerified.append((stoi[char]+3 if char in stoi else 2))
         numerified_chars.append([0] + [stoi_chars[x]+3 if x in stoi_chars else 2 for x in char])

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
       
         numerifiedCurrent = torch.LongTensor(numerifiedCurrent).view(args.batchSize, -1, sequenceLengthHere).transpose(0,1).transpose(1,2).cuda()
         numerifiedCurrent_chars = torch.LongTensor(numerifiedCurrent_chars).view(args.batchSize, -1, sequenceLengthHere, 16).transpose(0,1).transpose(1,2).cuda()

#         print(numerifiedCurrent_chars.size())
 #        quit()
         numberOfSequences = numerifiedCurrent.size()[0]
         for i in range(numberOfSequences):
             yield numerifiedCurrent[i], numerifiedCurrent_chars[i]
         hidden = None
       else:
         print("Skipping")

      print(("WASTED WORDS", len(numerified)))


hidden = None

zeroBeginning = torch.LongTensor([0 for _ in range(args.batchSize)]).cuda().view(1,args.batchSize)

zeroBeginning_chars = torch.zeros(1, args.batchSize, 16).long().cuda()


zeroHidden = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

bernoulli = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.1 for _ in range(args.batchSize)]).cuda())

bernoulli_input = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_in for _ in range(args.batchSize * 2 * args.word_embedding_size)]).cuda())
bernoulli_output = torch.distributions.bernoulli.Bernoulli(torch.tensor([1-args.weight_dropout_out for _ in range(args.batchSize * args.hidden_dim)]).cuda())


zeroChunk = torch.zeros((args.layer_num, args.batchSize, args.hidden_dim)).cuda()

hidden = None
beginning = zeroBeginning

def forward(numeric, train=True, printHere=False):
      global beginning
      global beginning_chars
      global hidden
      if not args.infinite_context or random.random() > 0.9:
          beginning = zeroBeginning
          beginning_chars = zeroBeginning_chars
          hidden = None
      elif hidden is not None:
        if args.celltype == "gru":
          hidden = Variable(hidden.data).detach()
        elif args.celltype == "lstm":
          hidden = tuple([Variable(x.data).detach() for x in hidden])

      numeric, numeric_chars = numeric

      numeric_noised = numeric

      numeric = torch.cat([beginning, numeric], dim=0)
      numeric_noised = torch.cat([beginning, numeric_noised], dim=0)

      beginning = numeric[-1:]
      input_tensor = Variable(numeric[:-1], requires_grad=False)
      target_tensor = Variable(numeric[1:], requires_grad=False)

      input_tensor_noised = Variable(numeric_noised, requires_grad=False)


      embedded = word_embeddings(input_tensor)
      if train:
         embedded = char_dropout(embedded)
         mask = bernoulli_input.sample()
         mask = mask.view(1, args.batchSize, 2*args.word_embedding_size)
         embedded = embedded * mask

      outputsSoFar = [] #(zeroChunk, zeroChunk)]
      fullOutputsSoFar = []
      result  = ["" for _ in range(args.batchSize)]
      retrieved = zeroChunk
      basAct = torch.zeros(1, args.batchSize).cuda() + 1
      fluctuatingActivations = [basAct]
      for i in range(args.sequence_length):
          embeddedLast = embedded[i].unsqueeze(0)
          if len(outputsSoFar) > 0:
             out_encoder_keys = torch.cat([x[0] for x in outputsSoFar], dim=0)
             out_encoder_values = torch.cat([x[1] for x in outputsSoFar], dim=0)
             attention_logits = attention_out(tanh(attention_proj(torch.cat([out_encoder_keys, out_decoder.expand(i, -1, -1), embeddedLast.expand(i, -1, -1)], dim=2))))

#             prior_activations = torch.stack(fluctuatingActivations, dim=2)
 #            powerDistances = torch.FloatTensor([(i-j) ** (-0.5) for j in range(0, i)]).cuda().unsqueeze(0).unsqueeze(0)
  #           multiplied = torch.log((prior_activations * powerDistances).sum(dim=2))
             attention_logits = attention_logits  #+ multiplied.unsqueeze(2)


             attention = attention_softmax(attention_logits)
             fluctuatingActivations.append(attention.squeeze(2))
             fluctuatingActivations = [torch.cat([x, basAct], dim=0) for x in fluctuatingActivations]
             attention = attention #.transpose(0,1)
             #print(attention.size(), attention.sum(dim=0).mean(), attention.sum(dim=2).mean(), attention.sum(dim=1).mean())
             if printHere:
                print("=============")
 #               print("FROM PRIOR ATTENTIONS")
#                print(multiplied[0])
#                print(torch.log(fluctuatingActivations[:2,]))
                print(attention.size(), attention.sum(dim=0).mean())
                print("ATTENTION LOGITS")
                print(attention_logits[:,0].view(-1))
                print("ATENTION")
                print(attention[:,0].view(-1), attention.size(), i)
                print(out_encoder_values.size(), attention.size())
             from_encoder = (out_encoder_values.unsqueeze(2) * attention.unsqueeze(3)).sum(dim=0).transpose(0,1)
             retrieved = from_encoder
          else:
             retrieved = toValues(zeroChunk)



          
          out_decoder, hidden = rnn_decoder(embeddedLast, retrieved)

          outputsSoFar.append((toKeys(hidden), toValues(hidden)))     # for retrieval
#          outputsSoFar.append((out_decoder, out_decoder))     # for retrieval

          fullOutputsSoFar.append(out_decoder) # for prediction

      out_decoder = torch.cat(fullOutputsSoFar, dim=0)

      if train:
        mask = bernoulli_output.sample()
        mask = mask.view(1, args.batchSize, args.hidden_dim)
        out_decoder = out_decoder * mask



      logits = output(out_decoder) 
      log_probs = logsoftmax(logits)
      loss = train_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1))

      if printHere:
         lossTensor = print_loss(log_probs.view(-1, len(itos)+3), target_tensor.view(-1)).view(-1, args.batchSize)
         losses = lossTensor.data.cpu().numpy()
         numericCPU = numeric.cpu().data.numpy()

         print(("NONE", itos_total[numericCPU[0][0]]))
         for i in range((args.sequence_length)):
            print((losses[i][0], itos_total[numericCPU[i+1][0]]))






      return loss, target_tensor.view(-1).size()[0]

def backward(loss, printHere):
      optim.zero_grad()
      if printHere:
         print(loss)
      loss.backward()
      torch.nn.utils.clip_grad_value_(parameters_cached, 5.0) #, norm_type="inf")
      optim.step()


lossHasBeenBad = 0
failedRuns = 0

import time

totalStartTime = time.time()

lastSaved = (None, None)
devLosses = []
for epoch in range(10000):
   print(epoch)
   training_data = corpusIterator_PTB_Deps.training(args.language)
   print("Got data")
   training_chars = prepareDatasetChunks(training_data, train=True)



   rnn_decoder.train(True)

   startTime = time.time()
   trainChars = 0
   counter = 0
   hidden, beginning = None, zeroBeginning
   while True:
      counter += 1
      try:
         numeric = next(training_chars)
      except StopIteration:
         break
      printHere = (counter % 50 == 0)
      loss, charCounts = forward(numeric, printHere=printHere, train=True)
      backward(loss, printHere)
      if loss.data.cpu().numpy() > 15.0:
          lossHasBeenBad += 1
      else:
          lossHasBeenBad = 0
      if lossHasBeenBad > 100:
          print("Loss exploding, has been bad for a while")
          print(loss)
          quit()
      trainChars += charCounts 
      if printHere:
          print(("Loss here", loss))
          print((epoch,counter, failedRuns))
          print("Dev losses")
          print(devLosses)
          print("Words per sec "+str(trainChars/(time.time()-startTime)))
          print(learning_rate)
          print(lastSaved)
          print(__file__)
          print(args)
#      if counter % 2000 == 0: # and epoch == 0:
#        state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
#        torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")
#        lastSaved = (epoch, counter)
      if (time.time() - totalStartTime)/60 > 4000:
          print("Breaking early to get some result within 72 hours")
          totalStartTime = time.time()
          break

 #     break
   rnn_decoder.train(False)


   dev_data = corpusIterator_PTB_Deps.dev(args.language)
   print("Got data")
   dev_chars = prepareDatasetChunks(dev_data, train=False)


     
   dev_loss = 0
   dev_char_count = 0
   counter = 0
   hidden, beginning = None, zeroBeginning
   while True:
       counter += 1
       try:
          numeric = next(dev_chars)
       except StopIteration:
          break
       printHere = (counter % 50 == 0)
       loss, numberOfCharacters = forward(numeric, printHere=printHere, train=False)
       dev_loss += numberOfCharacters * loss.cpu().data.numpy()
       dev_char_count += numberOfCharacters
   devLosses.append(dev_loss/dev_char_count)
   print(devLosses)

   with open("/u/scr/mhahn/recursive-prd/actr/estimates-"+args.language+"_"+__file__+"_model_"+str(args.myID)+".txt", "w") as outFile:
       print(str(args), file=outFile)
       print(" ".join([str(x) for x in devLosses]), file=outFile)

   if devLosses[-1] > 10:
      break

   if len(devLosses) > 1 and devLosses[-1] > min(devLosses):
      failedRuns += 1
      learning_rate = learning_rate * args.lr_decay_after_failure

   if failedRuns > 10:
      break
#   state = {"arguments" : str(args), "words" : itos, "components" : [c.state_dict() for c in modules]}
#   torch.save(state, "/u/scr/mhahn/CODEBOOKS/"+args.language+"_"+__file__+"_code_"+str(args.myID)+".txt")
#   lastSaved = (epoch, counter)






   learning_rate = learning_rate * args.lr_decay
   optim = torch.optim.SGD(parameters(), lr=learning_rate, momentum=0.0) # 0.02, 0.9


