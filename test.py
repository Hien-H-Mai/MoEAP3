import time
import numpy as np
from GAPrompt.arguments import get_args
from GAPrompt.modelwrapper import SampleModelWrapper, FitnessModelWrapper, label_to_token_id
from GAPrompt.petri import PetriDish, init_population
from GAPrompt.data_utils import load_raw_data, SET_LABEL_POSITION
from GAPrompt.utils import set_seed, list_skip_none

from task_config import *

import warnings
import os
import json
warnings.filterwarnings('ignore') 

class Verifier:
    def __init__(self,tokenizer,max_length,keys) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keys = keys

    def verify(self,example):
        full_text = ''
        for i,k in enumerate(self.keys):
            if i == 0:
                full_text += str(example[k]) 
            else:
                full_text += ' '+str(example[k])
        features = self.tokenizer(full_text)
        return len(features['input_ids']) < self.max_length 

class Cutter:
    def __init__(self,tokenizer,max_length,keys) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keys = keys

    def cut(self,example):
        full_text = ''
        for i,k in enumerate(self.keys):
            if i == 0:
                full_text += str(example[k]) 
            else:
                full_text += ' '+str(example[k])
        features = self.tokenizer(full_text)
        full_len = len(features['input_ids'])
        if full_len >= self.max_length:
            longest = 0
            key_longest = None
            for k in self.keys:
                if k == 'label':
                    continue
                if len(example[k]) > longest:
                    longest = len(example[k])
                    key_longest = k
            
            text_to_cut = example[key_longest]
            features = self.tokenizer(text_to_cut,add_special_tokens=False)
            new_text = self.tokenizer.decode(features['input_ids'][:self.max_length-full_len-1])
            example[key_longest] = new_text
        return example

def cut_long_sentences(raw_dataset,cutter):
    new_dataset = []
    for example in raw_dataset:
        new_dataset.append(cutter.cut(example))
    return new_dataset

def load_data_as_text_chunks(data,keys,labels):
    ans = []
    for i in range(len(data)):
        example = []
        for x in keys:
            s = data[i][x]
            if labels and isinstance(s,int):
                s = labels[s]
            example.append(s)
        ans.append(example)
    return ans

def construct_true_few_shot_data(task_name, split, k_shot, verifiers=None):
    
    train_label_count = {}
    dev_label_count = {}
    new_train_data = []
    new_dev_data = []

    train_data = load_raw_data(task_to_names[task_name],split,task_to_datafiles[task_name])

    all_indices = [_ for _ in range(len(train_data))]
    np.random.shuffle(all_indices)

    for index in all_indices:
        label = train_data[index]['label']
        if label < 0:
            continue

        if label not in train_label_count:
            train_label_count[label] = 0
        if label not in dev_label_count:
            dev_label_count[label] = 0

        if train_label_count[label] < k_shot:
            #if verifier and not verifier.verify(train_data[index]):
            if verifiers:
                skip = False
                for verif in verifiers:
                    if not verif.verify(train_data[index]):
                        skip = True
                if skip:
                    continue
            new_train_data.append(train_data[index])
            train_label_count[label] += 1
        elif dev_label_count[label] < k_shot:
            if verifiers:
                skip = False
                for verif in verifiers:
                    if not verif.verify(train_data[index]):
                        skip = True
                if skip:
                    continue
            new_dev_data.append(train_data[index])
            dev_label_count[label] += 1
    
    train_text_chunks = load_data_as_text_chunks(new_train_data,task_to_keys[task_name],task_to_labels[task_name])
    dev_text_chunks = load_data_as_text_chunks(new_dev_data,task_to_keys[task_name],task_to_labels[task_name])

    return train_text_chunks, dev_text_chunks

def check_hyper_params(template,args):
    length = 0
    for e in template:
        if e is not None:
            length += 1
    if length == 1:
        args.crossover_prob = 0
        args.mutate_prob = 1

def move_labels_to_the_end(label_list):
    label_list.remove('label')
    label_list.append('label')
    return label_list

args = get_args()

set_seed(args.seed)

if 'gpt' in args.eval_model:
    task_to_keys[args.task_name] = move_labels_to_the_end(task_to_keys[args.task_name])

SET_LABEL_POSITION(task_to_keys[args.task_name].index('label'))

check_hyper_params(task_to_prompt_templates[args.task_name],args)

#batch_size = min(args.k_shot * len(task_to_labels[args.task_name]), 64)


batch_size = args.batch_size
    
device = None
    
fitness_model = FitnessModelWrapper(args.eval_model,max_length=args.max_seq_length,batch_size=batch_size, device=device)

if task_to_metric[args.task_name] != None:
    metric_class, pos_label_index = task_to_metric[args.task_name]
    pos_label = label_to_token_id(task_to_labels[args.task_name][pos_label_index],fitness_model.tokenizer)
    fitness_model.set_custom_metric( metric_class(pos_label) )


raw_test_data = load_raw_data(
        task_to_names[args.task_name],
        task_to_dataset_tags[args.task_name][1],
        task_to_datafiles[args.task_name],
    )

test_data = load_data_as_text_chunks(
        raw_test_data,
        task_to_keys[args.task_name],
        task_to_labels[args.task_name],
    )

result_file = open(args.test_prompt_file, "r") 
        
results = [json.loads(jline) for jline in result_file.read().splitlines()]
from GAPrompt.individual import Individual
    
from copy import deepcopy
individuals = []
for res in results:
    
    ind = Individual(chromosomes=res["chromosomes"])
    
    ind.prompts = deepcopy(res["prompts"])
    
    ind.ppl = res["ppl"]
    
    individuals.append(ind)
petri = PetriDish(test_data,individuals,None,fitness_model,None,args)
# print(individuals)
print('\n# of test data: {}'.format(len(test_data)), flush=True)
print('Example:', flush=True)
print(test_data[0], flush=True)
# petri.__print_individuals()
start = time.time()
print("Start to test")
    
petri.test(individuals,test_data)
    
print("Test time:", (time.time() - start)/60, "minutes")
petri.save_individuals("final_test")
for i,ind in enumerate(petri.individuals):
    petri.log("{} : {} > {} {}".format(i,' ||| '.join(list_skip_none(ind.prompts)),ind.fitness_score,ind.accuracy),flush=True)
# petri.__print_individuals()
