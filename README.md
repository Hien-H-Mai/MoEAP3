Source code for the paper: [On the Investigation of Evolutionary Multi-Objective Optimization for Discrete Prompt Search](https://openreview.net/forum?id=tbJg5QXjuuY)


#### Requirements
`pip install -r requirements.txt`

#### Quick Start
1. For fine-tuning DeBERTa-v3 on CoLA dataset, run cell in `glue_ft.ipynb`. After fine-tuning, Let's put checkpoint, tokenizer, config files in `cola-deberta-v3-large` folder
2. For RoBERTa-based experiments, run the scripts via `bash run_roberta.sh [random_seed] [task_name]`.
3. For GPT2-based experiments, run the scripts via `bash run_gpt.sh [random_seed] [task_name]`.    
3. For FlanT5 (or T5) -based experiments, run the scripts via `bash run_t5.sh [random_seed] [task_name]`.   
4. Important arguments:
   * `--task_name`: The name of a task. choices = `[mrpc, snli, sst2, rte, mr, agnews, dbpedia]`.
   * `--k_shot`: number of shots.
   * `--petri_size`: number of individuals in each generation.
   * `--petri_iter_num`: number of generations/iterations.
   * `--mutate_prob`: token mutation probability.
   * `--crossover_prob`: chromosome crossover probability.
5. For test all individuals in population (for Pareto front), run `test.py`

#### Reference
This code is inspired by [GAP3](https://www.ijcai.org/proceedings/2023/0588.pdf) and [Pymoo](https://pymoo.org/)
   