''''NOTE: This is currently set to get the indices STARTING FROM 1.'''

import sys
import argparse
import pandas as pd
from pathlib import Path
import yaml



with open(Path(__file__).parents[1] / "katie_paths.yaml", 'r') as file:
   paths = yaml.safe_load(file)

REPO_ROOT = paths['repo_root']
input_conlls = {"Dev":f"{REPO_ROOT}/training-data/ptb3-wsj-dev.conllx","Test":f"{REPO_ROOT}/training-data/ptb3-wsj-test.conllx","Train":f"{REPO_ROOT}/training-data/ptb3-wsj-train.conllx"}



bignonos = ('advcl','ccomp','csubj','csubjpass','dep','mark','parataxis','pcomp','rcmod','ref','vmod','xcomp','xsubj')
smallnonos = ('appos','discourse','mwe')
optnonos = ('auxpass','nsubjpass','cc','conj','neg')
sent_final_punct = ('.', '!') ##Questions are currently excluded


file_records = {}
# x = 1671, 1677

# for line in open(args.input_conll_filepath):
for input_set in input_conlls.keys():
    print(f"Starting {input_set}")
    counter = 0
    sent_idx = 0
    sent = ['']
    nsubj_counter = 0 
    skip_sent = False
    records = []
    # records2 = []
    print(input_conlls[input_set])
    for line in open(input_conlls[input_set]): ## Iterate through the lines
        if sent_idx % 10000 == 0:
            print("Starting sentence", sent_idx)
        if line.startswith('#'): ##Not sure what this symbol means; doesn't show up in dev
            print("TF IS THIS", line, sent_idx)
            counter += 1
            continue
        if not line.strip(): ##blank line indicates the end of prev sent and splits prev from next sentence
            if nsubj_counter == 0: ## This is to avoid sentence fragments without subjects
                sent_idx +=1 ##moves on to next sent
                nsubj_counter = 0 ##sets nsubj counter to 0
                skip_sent = False ##resets to not skip next sent
                sent[0] = '' ##Reset saved sent to nothing (avoids back-to-back applicable sents)
            else:
                sent_idx +=1
                nsubj_counter = 0
                skip_sent = False
                stripped_sent = sent[0].lstrip() ##This is to strip the '' that is sent[0] for all sents
                if stripped_sent != '':
                    split_sent = stripped_sent.split(" ")
                    for punc in sent_final_punct: ##OptA: Use this to make sure there's a sent_final_punc in the line (not true for headlines)
                        if punc in split_sent: 
                            # records.append((sent_idx, final))
                            records.append((sent_idx,stripped_sent))
                            break
                    # records.append((sent_idx,stripped_sent)) ##OptB: Use this if you want to include headlines (some are fragments, eg idx 14987,30564)
                sent[0] = '' ##Reset saved sent to nothing (avoids back-to-back applicable sents)
        else:
            wordidx0 = int(line.split('\t')[0])
            wordidx1 = int(line.split('\t')[0]) + 1
            word = line.split('\t')[1]
            udpos = line.split('\t')[3]
            pennpos = line.split('\t')[4]
            deplab = line.split('\t')[7]
            headidx = line.split('\t')[6]


            if skip_sent == False:
                if deplab in bignonos:
                    sent[0] = ''
                    skip_sent = True
                    continue
                elif nsubj_counter >1:
                    sent[0] = ''
                    skip_sent = True
                    continue
                elif deplab in smallnonos:
                    sent[0] = ''
                    skip_sent = True
                    continue
                elif deplab in optnonos:
                    sent[0] = ''
                    skip_sent = True
                    continue    
    
                else:
                    if deplab == 'nsubj':
                        nsubj_counter += 1
                    elif deplab == 'punct': ## This is to get rid of any non-sent-final punctuation
                        if word not in sent_final_punct:
                            sent[0] = ''
                            skip_sent = True
                            continue 
                    sent_so_far = sent[0]
                    curr_sent_so_far = sent_so_far + f' {word}'
                    sent[0] = curr_sent_so_far
            else:
                continue
                
    file_records[input_set] = records


##To write to file:
with open (f"{REPO_ROOT}/scripts/simple_sents.txt", "w") as file:
    for input_set in file_records.keys():
        file.write(f"{input_set}\n")
        for record in file_records[input_set]:
            file.write(f"{record[0]}\t{record[1]}\n")
        file.write(f"Number of {input_set} Sents: {len(file_records[input_set])}\n\n")

##To just get sent_idx
for input_set in file_records.keys():
    print(f"{input_set}\n")
    idx_1 = []
    idx_0 = []
    for record in file_records[input_set]:
        idx_1.append(record[0])
        idx_0.append(record[0]-1)
    # print(f"Sent Idx (starting from 1): {idx_1}") ##Uncomment if you want from 1
    print(f"Sent Idx (starting from 0): {idx_0}")