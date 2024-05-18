from .data_utils import *
from .utils import *
import random
import math
from tqdm import tqdm
from .individual import Individual
from .modelwrapper import SampleModelWrapper, FitnessModelWrapper
import copy
import logging
from .thread_utils import SampleTokenWorker, FitWorker
from threading import Thread
from queue import Queue
import torch
from .thread_utils import global_cache
from .moo_utils import RankAndCrowding
import time
from functools import cmp_to_key
from pymoo.util.dominator import Dominator
# from evaluate import load
import json
from pymoo.operators.selection.tournament import compare
from .quality_score import Grammar
import numpy as np
def init_population(template,n_individuals,mask_token_id):
    population = []
    print("Sample {} individuals as the initial population...".format(n_individuals),flush=True)
    #cache = Cache()
    #while len(population) < n_individuals:
    for _ in range(n_individuals):
        chromosomes = copy.deepcopy(template)
        seed_chr = random_choice_skip_none(chromosomes)
        #seed_chr = random.choice(chromosomes)
        #while seed_chr is None:
        #    seed_chr = random.choice(chromosomes)
        seed_chr.append(mask_token_id)
        ind = Individual(chromosomes,True)
        population.append(ind)
        #cache.add(ind)
    return population
def better_than(ind1,ind2):
    if math.fabs(ind1.accuracy - ind2.accuracy) < 1e-8:
        return ind1.fitness_score - ind2.fitness_score
    else:
        return ind1.accuracy - ind2.accuracy
# def better_than(ind1,ind2):
#     if math.fabs(ind1.accuracy + ind1.ppl - ind2.accuracy - ind2.ppl) < 1e-8:
#         return ind1.fitness_score + ind1.ppl - ind2.fitness_score - ind2.ppl
#     else:
#         return ind1.accuracy + ind1.ppl - ind2.accuracy - ind2.ppl
class PetriDish:
    def __init__(self,
        raw_train_data,
        #raw_dev_data,
        individuals,
        mlm_model,
        eval_model,
        
        moo_mode,
        args,
    ):
        self.raw_train_data = raw_train_data
        if raw_train_data and mlm_model:
            self.data_sample = tokenize_data_chunks(raw_train_data,mlm_model.tokenizer)
        #self.raw_dev_data = raw_dev_data
        self.individuals = individuals
        self.mlm_model = mlm_model
        self.eval_model = eval_model
        #self.cache = Cache()
        self.best_individual = None
        #self.log_file_handler = None
        self.logger = None
        self.args = args
        self.to_verify_on_dev = []
        
        self.tournament_type = "comp_by_dom_and_crowding"
        
        self.RankAndCrowding = RankAndCrowding()
        
        self.moo_mode = moo_mode
        
        # self.perplexity = load("perplexity", module_type="metric")
        if moo_mode is not None:
            self.quality = Grammar()
    
    def set_log_file(self,filename):
        log_file_handler = logging.FileHandler(filename)
        #f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #log_file_handler.setFormatter(f_format)
        self.logger = logging.getLogger(filename)
        self.logger.addHandler(log_file_handler)
    def close_log(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.flush()
            handler.close()
    def release(self):
        self.close_log()
        self.raw_train_data = None
        #self.raw_dev_data = None
        self.data_sample = None
        self.mlm_model = None
        self.eval_model = None
        #self.cache = None
        self.logger = None
    
    def log(self,string,flush=True):
        if self.logger == None:
            print(string,flush=flush)
        else:
            self.logger.critical(string)
    def print(self):
        for i,s in enumerate(self.raw_train_data):
            self.log("{}: {}".format(i,s),flush=True)
        for i,d in enumerate(self.data_sample):
            self.log("{}: {}".format(i,d),flush=True)
        #for i,s in enumerate(self.raw_dev_data):
        #    self.log("{}: {}".format(i,s),flush=True)
        self.__print_individuals()
    
    def __print_individuals(self):
        self.decode_prompts(self.individuals)
        for i,ind in enumerate(self.individuals):
            self.log("{} : {} > {} {} {}".format(i,' ||| '.join(list_skip_none(ind.prompts)),ind.fitness_score,ind.ppl,ind.accuracy),flush=True)
    def realize(self):
        n_threads = torch.cuda.device_count()
        to_realize_queue = Queue()
        for ind in self.individuals:
            if not ind.abstract:
                #population.append(ind)
                #self.cache.add(ind)
                #global_cache.add(ind)
                continue
            to_realize_queue.put(ind)
        #print("{} to realize".format(to_realize_queue.qsize()),flush=True)
        for i in range(n_threads):
            worker = SampleTokenWorker(to_realize_queue,i,self.mlm_model,self.data_sample,self.args)
            worker.daemon = True
            worker.start()
        to_realize_queue.join()
        #torch.cuda.empty_cache()
    def __label_statistics(self):
        def stat_dataset(dataset, name):
            
            counts = {}
            for example in dataset:
                label = example[GET_LABEL_POSITION()]
                if label not in counts.keys():
                    counts[label] = 1
                else:
                    counts[label] += 1
            self.log("{} Set:".format(name),flush=True)
            for k, v in counts.items():
                self.log("{}:\t{}\t{}".format(k,v,v/len(dataset)),flush=True)
        stat_dataset(self.raw_train_data,'Train')
        #stat_dataset(self.raw_dev_data,'Dev')
    def save_individuals(self, gen = None):
        
        individuals = []
        
        for ind in self.individuals:
            
            data = dict()
            
            data["prompts"] = ind.prompts
            
            data["chromosomes"] = ind.chromosomes
            
            data["fitness_score"] = ind.fitness_score
            
            data["ppl"] = ind.ppl
            
            data["accuracy"] = ind.accuracy
            
            data["rank"] = ind.rank
            
            data["crowding"] = ind.crowding
            individuals.append(data)
        
        if gen is None:
            
            gen = "final"
            
        if gen == "final_test" and self.args.test_prompt_file is not None:
            
            save_name = self.args.test_prompt_file.replace("results","test_results")
            
        else:
            
            save_name = f'./{self.args.mlm_model.replace("/","-")}_{self.args.eval_model.replace("/","-")}_{self.args.task_name}_{self.args.seed}_{self.args.algo}_{gen}_results.jsonl'
        with open(save_name, "w") as file:
            
            for entry in individuals:
                json.dump(entry, file)
                file.write('\n')
        
    
    def evolute(self, crossover_prob, mutate_prob, n_generations, debug=False):
        n_population = len(self.individuals)
        n_elites = int(math.sqrt(n_population))
        self.log("Label distribution:",flush=True)
        self.__label_statistics()
        
        self.realize()
        self.eval()
            
        
        
        #early_stopping_count = 0
        for i in tqdm(range(n_generations),desc ="Evolutions"):
            start_time = time.time()
            #if debug:
            self.log("Iteration {}:".format(i),flush=True)
            self.__print_individuals()
            elites = self.top(n_elites)
            if self.best_individual == None or self.best_individual.fitness_score == None or better_than(elites[0],self.best_individual):
                self.best_individual = elites[0]
                #early_stopping_count = 0
            else:
                elites.append(self.best_individual)
                #early_stopping_count += 1
            
            #if i >= self.args.warmup and early_stopping_count >= 2:
            #    self.log("Early stopping reached!!!")
            #    break
            
            
            # for ind in self.individuals:
            #     print("##")
            #     for chr in ind.chromosomes:
            #         print(self.mlm_model.tokenizer.decode(chr))
            
            if i >= self.args.warmup:
                self.to_verify_on_dev += elites[:self.args.top_k_dev]
            
            new_generation = []
            max_steps = n_population*10
            iter = 0
            print("Length of init population:", len(elites))
            while len(new_generation) < n_population and iter < max_steps:
                individual1, individual2 = sample_pair(elites,True)
                new_individuals = individual1.crossover(individual2, crossover_prob)
                if debug:
                    self.log("Crossover: ",flush=True)
                    self.log(individual1.chromosomes,flush=True)
                    self.log(individual2.chromosomes,flush=True)
                    self.log("Obtained:",flush=True)
                    for x in new_individuals:
                        self.log(x.chromosomes)
                for x in new_individuals:
                    if debug:
                        self.log("Mutate: ",flush=True)
                        self.log(x.chromosomes,flush=True)
                    x.mutate(mutate_prob,self.mlm_model.tokenizer.mask_token_id)
                    if debug:
                        self.log("Obtained:",flush=True)
                        self.log(x.chromosomes,flush=True)
                    #if not self.cache.is_cached(x):
                    #    new_generation.append(x)
                    #    self.cache.add(x)
                    #elif debug:
                    #    self.log('cached',flush=True)
                    if not x.abstract:
                        if global_cache.is_cached(x):
                            continue
                        global_cache.add(x)
                    
                    new_generation.append(x)
                iter += 1
            
            #if len(new_generation) < n_population:
            #    self.individuals = self.individuals + new_generation
            #    self.individuals = self.top(n_population,keep_the_best=False)
            #else:
            print("Length of new population:", len(new_generation))
            self.individuals = new_generation
            self.realize()
            self.eval()
            
            end_time = time.time()
            self.log("--- %s seconds ---" % (end_time - start_time),flush=True)
            self.save_individuals(gen = i)
        #self.log('>>>Best score: {} accuracy: {}'.format(self.best_individual.fitness_score,self.best_individual.accuracy),flush=True)
        #self.log('>>>Prompt: '+' ||| '.join(self.best_individual.prompts),flush=True)
        last_best = self.top(1)[0]
        if self.best_individual == None or self.best_individual.fitness_score == None or better_than(last_best,self.best_individual):
            self.best_individual = last_best
        self.save_individuals()
    def top(self,k):
        if k > len(self.individuals):
            k = len(self.individuals)
        self.individuals = sorted(self.individuals, key=cmp_to_key(better_than), reverse=True)
        
        return self.individuals[:k]
    
    def moo_evolute(self, crossover_prob, mutate_prob, n_generations, debug=False):
        
        n_population = len(self.individuals)
        n_elites = int(math.sqrt(n_population))
        self.log("Label distribution:",flush=True)
        self.__label_statistics()
        #early_stopping_count = 0
        self.realize()
        self.eval()
        
        
        
        self.RankAndCrowding.do(problem=None,
            
                               pop=self.individuals,
                               
                               return_indices=False,
                               
                               n_survive = len(self.individuals))
        for i in tqdm(range(0, n_generations),desc ="Evolutions"):
            start_time = time.time()
            #if debug:
            self.log("Iteration {}:".format(i),flush=True)
            
            self.__print_individuals()
            elites = self.top(n_elites)
            if self.best_individual == None or self.best_individual.fitness_score == None or better_than(elites[0],self.best_individual):
                self.best_individual = elites[0]
                #early_stopping_count = 0
            else:
                elites.append(self.best_individual)
                #early_stopping_count += 1
                
                
            if i >= self.args.warmup:
                self.to_verify_on_dev += elites[:self.args.top_k_dev]
            
            new_generation = []
            max_steps = n_population*10
            iter = 0
               
            selected_generation = self.moo_selection()
            if len(selected_generation) != self.args.petri_size:
                raise ValueError(f"The size of selected individuals is {len(selected_generation)} == size of population {self.args.petri_size}")
            
            random.shuffle(selected_generation)
            for ind in self.individuals:
                ind.abstract = False
            print("Length of init population:", len(self.individuals))
            
            while  iter < max_steps and len(new_generation) < len(selected_generation):
                # print(iter)
                individual1 = selected_generation[iter]
                try:
                    individual2 = selected_generation[iter+1]
                except:
                    individual2 = selected_generation[0]
                new_individuals = individual1.crossover(individual2, crossover_prob)
                if debug:
                    self.log("Crossover: ",flush=True)
                    self.log(individual1.chromosomes,flush=True)
                    self.log(individual2.chromosomes,flush=True)
                    self.log("Obtained:",flush=True)
                    for x in new_individuals:
                        self.log(x.chromosomes)
                for x in new_individuals:
                    if debug:
                        self.log("Mutate: ",flush=True)
                        self.log(x.chromosomes,flush=True)
                    x.mutate(mutate_prob,self.mlm_model.tokenizer.mask_token_id)
                    if debug:
                        self.log("Obtained:",flush=True)
                        self.log(x.chromosomes,flush=True)
                    #if not self.cache.is_cached(x):
                    #    new_generation.append(x)
                    #    self.cache.add(x)
                    #elif debug:
                    #    self.log('cached',flush=True)
                    if not x.abstract:
                        if global_cache.is_cached(x):
                            continue
                        global_cache.add(x)
                    
                    new_generation.append(x)
                iter += 1
            
            #if len(new_generation) < n_population:
            #    self.individuals = self.individuals + new_generation
            #    self.individuals = self.top(n_population,keep_the_best=False)
            #else:
            
            print("Length of new population:", len(new_generation))
            self.individuals.extend(new_generation)
            self.realize()
            self.eval()
            
            
            
            self.individuals = self.RankAndCrowding.do(problem=None,
            
                                                       pop=self.individuals,
                                                       
                                                       return_indices=False,
                                                       
                                                       n_survive = self.args.petri_size)
            if len(self.individuals) != self.args.petri_size:
                raise ValueError(f"The size of selected individuals is {len(self.individuals)} == size of population is {self.args.petri_size}")
            
            for ind in self.individuals:
                if not isinstance(ind, Individual):
                    print("!!!!!!!!!! The individual is", type(ind))
                    
            end_time = time.time()
            self.log("--- %s seconds ---" % (end_time - start_time),flush=True)
            
            self.save_individuals(gen = i)
        #self.log('>>>Best score: {} accuracy: {}'.format(self.best_individual.fitness_score,self.best_individual.accuracy),flush=True)
        #self.log('>>>Prompt: '+' ||| '.join(self.best_individual.prompts),flush=True)
        last_best = self.top(1)[0]
        if self.best_individual == None or self.best_individual.fitness_score == None or better_than(last_best,self.best_individual):
            self.best_individual = last_best
        self.save_individuals()
    def moo_selection(self, tournament_size = 2):
        """Tournament selection for MOO"""
        
        selected_individuals = []
        
        for _ in range(2):
            
            random.shuffle(self.individuals)
            
            for i in range(0, len(self.individuals), 2):
                
                a_f = [-1.0 * self.individuals[i].fitness_score, -1.0 * self.individuals[i].ppl]
                
                b_idx = i + 1 if i != len(self.individuals) - 1 else 0
                
                # if b_idx == 0:
                    
                b_f = [-1.0 * self.individuals[b_idx].fitness_score, -1.0 * self.individuals[b_idx].ppl]
                    
                # else:
                    
                #     b_f = [self.individuals[i + 1].fitness_score,self.individuals[i + 1].ppl]
                
                selected_ind = None
                
                if self.tournament_type == 'comp_by_dom_and_crowding':
                    
                    rel = Dominator.get_relation(a_f, b_f)
                    
                    if rel == 1:
                        
                        selected_ind = self.individuals[i]
                            
                    elif rel == -1:
                        
                        selected_ind = self.individuals[b_idx]
                elif self.tournament_type == 'comp_by_rank_and_crowding':
                    
                    raise NotImplementedError()
                
                    # S[i] = compare(a, rank_a, b, rank_b, method='smaller_is_better')
                else:
                    
                    raise Exception("Unknown tournament type.")
                # if rank or domination relation didn't make a decision compare by crowding
                if selected_ind is None:
                    
                    cd_a = self.individuals[i].crowding
                    
                    cd_b = self.individuals[b_idx].crowding
                    
                    selected_ind = compare(self.individuals[i], cd_a, self.individuals[b_idx], cd_b, method='larger_is_better', return_random_if_equal=True)
                    
                selected_individuals.append(selected_ind)
        return selected_individuals
            
    def verify_on_dev(self,raw_dev_data):
        self.decode_prompts(self.to_verify_on_dev)
        individuals_to_eval = Queue()
        for ind in self.to_verify_on_dev:
            individuals_to_eval.put(ind)
        n_threads = torch.cuda.device_count()
        for i in range(n_threads):
            worker = FitWorker(individuals_to_eval,i,self.eval_model,raw_dev_data)
            worker.daemon = True
            worker.start()
        individuals_to_eval.join()
    def best_on_dev(self,raw_dev_data):
        self.verify_on_dev(self.raw_train_data + raw_dev_data)
        rank_list = sorted(self.to_verify_on_dev, key=cmp_to_key(better_than), reverse=True)
        return rank_list[0]
    def decode_prompts(self,individuals):
        if not self.mlm_model:
            return
        for ind in individuals:
            ind.prompts = []
            for chr in ind.chromosomes:
                if chr is None:
                    ind.prompts.append(None)
                else:
                    prompt = self.mlm_model.tokenizer.decode(chr)
                    #if prompt.startswith('##'):
                    #    prompt = prompt[2:]
                    ind.prompts.append(prompt)
        
    def eval(self):
        self.decode_prompts(self.individuals)
        individuals_to_eval = Queue()
        need_fit = False
        
        individuals_to_eval_ppl = []
        for ind in self.individuals:
            if ind.fitness_score == None:
                individuals_to_eval.put(ind)
                need_fit = True
            
            if ind.ppl is None:
                
                individuals_to_eval_ppl.append(ind)
                
        if len(individuals_to_eval_ppl) != 0:
            
            try:
            
                prompt_1 = [ind.prompts[0] + "." for ind in individuals_to_eval_ppl]
                
            except:
                
                prompt_1 = None
            
            try:
                prompt_2 = [ind.prompts[1] + "." for ind in individuals_to_eval_ppl]
            
            except:
                
                prompt_2 = [ind.prompts[2] + "." for ind in individuals_to_eval_ppl]
                
                
            prompt_2_quality = self.quality.compute(predictions=prompt_2)
            
            if prompt_1 is not None:
            
                prompt_1_quality = self.quality.compute(predictions=prompt_1)
                
            else:
                
                prompt_1_quality = copy.deepcopy(prompt_2_quality)
                
            prompt_quality = (np.array(prompt_1_quality) + np.array(prompt_2_quality)) / 2.0
            prompt_quality = prompt_quality.tolist()
            
            i = 0
            
            for ind in self.individuals:
                
                if ind.ppl is None:
                
                    ind.ppl = prompt_quality[i] # acceptable
                    
                    i += 1
        if need_fit: 
            n_threads = torch.cuda.device_count()
            for i in range(n_threads):
                worker = FitWorker(individuals_to_eval,i,self.eval_model,self.raw_train_data)
                worker.daemon = True
                worker.start()
            individuals_to_eval.join()
    def test(self,individuals,raw_test_data, ppl_test = False):
        individuals_to_eval = Queue()
        
        individuals_to_eval_ppl = []
        for ind in individuals:
            individuals_to_eval.put(ind)
            
            individuals_to_eval_ppl.append(ind)
                
        n_threads = torch.cuda.device_count()
        for i in range(n_threads):
            worker = FitWorker(individuals_to_eval,i,self.eval_model,raw_test_data, True)
            worker.daemon = True
            worker.start()
        individuals_to_eval.join()
        
        
        
    def tpu_test(self,individuals,raw_test_data):
        
        print("******** TPU ********")
        
        self.eval_model.set_tpu_device()
        self.eval_model.fitness(individuals, raw_test_data, tpu=True)
class PetriDishGenerator:
    def __init__(self,
        raw_train_set,
        keys,
        labels,
        mlm_model,
        eval_model,
        args,
    ):
        self.raw_dataset = load_data_as_text_chunks(
            raw_train_set,
            keys,
            labels,
            data_size_for_debug=args.data_size_for_debug
        )
        self.mlm_model = mlm_model
        self.eval_model = eval_model
        self.args = args
    def generate_petri_dishes(self,population):
        n_petri_dishes = len(population)//self.args.petri_size
        n_petri_size = len(self.raw_dataset)//n_petri_dishes
        random.shuffle(self.raw_dataset)
        random.shuffle(population)
        petri_dishes = []
        for i in tqdm(range(n_petri_dishes),desc="Generating petri dishes"):
            sub_pop = population[i*self.args.petri_size:(i+1)*self.args.petri_size]
            sub_data = self.raw_dataset[i*n_petri_size:(i+1)*n_petri_size]
            dish = PetriDish(sub_data,sub_pop,self.mlm_model,self.eval_model,args=self.args)
            petri_dishes.append(dish)
        return petri_dishes
