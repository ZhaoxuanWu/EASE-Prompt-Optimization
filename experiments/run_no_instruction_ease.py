import random
import torch
import numpy as np
import sys
import copy
import os
import json
cwd = os.getcwd()
sys.path.append(cwd)
from automatic_prompt_engineer import config, template
from experiments.data.instruction_induction.load_data import load_data
from experiments.evaluation.instruction_induction.exec_accuracy import exec_accuracy_evaluator
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer
from LlamaForMLPRegression import NeuralTSDiag

from tqdm import tqdm
import argparse
from misc import set_all_seed, TASKS
import time
import datetime
import ot

print(torch.cuda.is_available())
print(torch.cuda.device_count())
## bayesian opt
tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}

STOP_TRAINING_AFTER_ITER=2000

model_name = 'mpnet'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
api_model = 'chatgpt'

class ForwardAPI:
    def __init__(self, model_name='mpnet', HF_cache_dir=None, conf=None, base_conf=None, eval_data=None,
                 prompt_gen_data=None, few_shot_data=None):
        kwargs={'torch_dtype': torch.float32}
        if model_name in ['mpnet']:
            self.tokenizer = AutoTokenizer.from_pretrained(HF_cache_dir)
            self.model = AutoModel.from_pretrained(HF_cache_dir, **kwargs)
            
            self.model = self.model.cuda()
        else:
            raise NotImplementedError
        
        ## eval preparation
        self.conf = config.update_config(conf, base_conf)
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate("[full_DEMO]\n\nInput: [INPUT]\nOutput: [OUTPUT]")
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        if few_shot_data is None:
            self.few_shot_data = prompt_gen_data
        
        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.best_exemplars = None
        self.best_exemplar_indices = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        
    def get_hidden_states(self, sentences):
        # Mean Pooling - Take attention mask into account for correct averaging
        def mean_pooling(model_output, attention_mask):
            token_embeddings = model_output[0] #First element of model_output contains all token embeddings
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        # Tokenize sentences
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input.to('cuda'))

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        
        return sentence_embeddings
        
    def eval_exemplars(self, instruction, selected_indices, n_shot_data):
        self.num_call += 1
        
        print('Instruction: {}'.format(instruction))
        n_shot_text = self.demos_template.fill(n_shot_data)
        
        if (instruction[0] + n_shot_text) in self.prompts_set.keys():
            (dev_perf, instruction_score) = self.prompts_set[instruction[0] + n_shot_text]
        else:
            if api_model in ['chatgpt']: 
                dev_perf, instruction_score, model_outputs, answers = self.conf['evaluation']['method'](instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation'], given_demos=n_shot_data)
                dev_perf = dev_perf.sorted()[1][0]
                # print(answers)
                # import pdb; pdb.set_trace()
                self.prompts_set[instruction[0] + n_shot_text] = (dev_perf, instruction_score)
            elif api_model in ['vicuna']:
                dev_perf, instruction_score, model_outputs, answers = self.conf['evaluation']['method'](instruction, self.eval_template, self.eval_data, self.demos_template, self.few_shot_data, self.conf['evaluation'], model_api=self, given_demos=n_shot_data)
                dev_perf = dev_perf.sorted()[1][0]
                self.prompts_set[instruction[0] + n_shot_text] = (dev_perf, instruction_score)
            else:
                raise NotImplementedError

        if dev_perf >= self.best_last_perf:
            self.count += 1

        if dev_perf >= self.best_dev_perf:
            self.best_dev_perf = dev_perf
            self.best_instruction = instruction
            self.best_exemplar_indices = selected_indices
            self.best_exemplars = self.demos_template.fill(n_shot_data)

        print('Dev loss: {}. Dev perf: {}. Best dev perf: {}'.format(
            round(float(dev_perf), 4),
            round(float(dev_perf), 4),
            round(float(self.best_dev_perf), 4)))
        print('********* Done *********')

        return dev_perf, instruction_score, instruction[0]
    
    def return_best_prompt(self):
        return self.best_instruction
    
    def return_best_exemplars(self):
        return self.best_exemplars
    
    def return_best_exemplar_indices(self):
        return self.best_exemplar_indices

    def return_prompts_set(self):
        return self.prompts_set
    

####################################
    
def run(task, HF_cache_dir, lam, nu, max_prompt_gen_size, NN_training_iter, n_domain, gpt_model, num_shot, total_iter, n_init):
    
    test_data = load_data('eval', task)
    prompt_gen_data = load_data('induce', task)
    prompt_gen_size = min(len(prompt_gen_data[0]), max_prompt_gen_size)
    prompt_gen_data = (prompt_gen_data[0][:prompt_gen_size], prompt_gen_data[1][:prompt_gen_size])
    eval_data = load_data('valid', task)
    # Make sure the eval data are all the same across eval trials
    min_size = min(20, len(eval_data[0]))
    eval_data = (eval_data[0][:min_size], eval_data[1][:min_size])
    
    # Data is in the form input: single item, output: list of items
    # For prompt_gen_data, sample a single item from the output list
    prompt_gen_data = prompt_gen_data[0], [random.sample(output, 1)[0]
                                           for output in prompt_gen_data[1]]
    
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "[full_DEMO]\n\nInput: [INPUT]\nOutput: [OUTPUT]" # change the evaluation template

    base_conf = '../experiments/configs/instruction_induction.yaml'
    conf = {
        'generation': {
            'num_subsamples': 1,
            'num_demos': 5,
            'num_prompts_per_subsample': 20,
            'model': {
                'gpt_config': {
                    'model': 'vicuna',
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator,
            'task': task,
            'num_samples': min(20, len(eval_data[0])),
            # 'num_samples': min(5, len(eval_data[0])),
            'model': {
                'gpt_config': {
                    'model': gpt_model,
                }
            }
        }
    }

    d_template = template.DemosTemplate(demos_template)    
    print("Number of data points for instruction generation: {}".format(len(prompt_gen_data[0])))
        
    model_forward_api = ForwardAPI(model_name=model_name, HF_cache_dir=HF_cache_dir, conf=conf, 
                                   base_conf=base_conf, eval_data=eval_data, prompt_gen_data=prompt_gen_data)
    
    # We do not need the gradient for the weights
    for param in model_forward_api.model.parameters():
        param.requires_grad = False
    
    instruction_log = []
    r_log = []
    exemplars_log = []
    indices_log = []
    
    prompts = [""]
    
    with torch.no_grad():        
        
        # Generate initial indices for exemplars
        init_indices = []
        for _ in range(n_init):
            init_indices.append(np.random.choice(prompt_gen_size, num_shot, replace=False))
        init_indices = np.stack(init_indices)
        
        sentences = []
        for i in range(len(init_indices)):
            n_shot_data = ([prompt_gen_data[0][j] for j in init_indices[i]], [prompt_gen_data[1][j] for j in init_indices[i]])
            n_shot_text = d_template.fill(n_shot_data)
            sentences.append(n_shot_text)
                        
        X_train = model_forward_api.get_hidden_states(sentences)
        
        # Evaluate the initial points
        Y = []
        pbar = tqdm(range(n_init))
        for i in pbar:
            
            n_shot_data = ([prompt_gen_data[0][j] for j in init_indices[i]], [prompt_gen_data[1][j] for j in init_indices[i]])
            n_shot_text = d_template.fill(n_shot_data)
            
            r, _, instruction = model_forward_api.eval_exemplars(prompts, init_indices[i], n_shot_data)            
            print(init_indices[i])
            print(n_shot_text)
            indices_log.append(init_indices[i].tolist())
            exemplars_log.append(n_shot_text)
            instruction_log.append(instruction)
            r_log.append(r)
            Y.append(r)
            if r == 1:
                X_train = X_train[:len(Y)]
                break
        y_train = torch.FloatTensor(Y).unsqueeze(-1).to(**tkwargs)
        print(f"Best initial point: {y_train.max().item():.3f}")    

    best_r = y_train.max().item()
    best_init = best_r
    max_iter = total_iter - n_init

    # Start bayesian opt

    # lambda represent regularization, nu represent the trade-off, grid search
    l = NeuralTSDiag(input_dim=X_train.shape[1], lamdba=lam, nu=nu, init_x=X_train, init_y=y_train, style='ucb', diagonalize=True)
    l.train(None, None, NN_training_iter)

    # precompute the representation for all the examplar sentences
    all_examplars_sentences = [d_template.fill(([prompt_gen_data[0][i_]],[prompt_gen_data[1][i_]])) for i_ in range(len(prompt_gen_data[0]))]
    all_examplars_representation = model_forward_api.get_hidden_states(all_examplars_sentences)
    
    # eval examplars representation
    eval_examplars_representation = model_forward_api.get_hidden_states([d_template.fill(([eval_data[0][i]],[eval_data[1][i][0]])) for i in range(len(eval_data[0]))])
    for t in range(max_iter):
        if r == 1:
            break
        
        with torch.no_grad():
            start_time = time.time()
            # Sample combinations
            N_SAMPLE = n_domain
            # N_SAMPLE = 10000
            
            batch_size = 50
            num_batches = N_SAMPLE // batch_size
            
            top_score = 0
            N_TIMES_DOMAIN = 50
            for _ in range(num_batches):
                sampled_indices = torch.ones(batch_size * N_TIMES_DOMAIN, prompt_gen_size).multinomial(num_shot, replacement=False).cuda()
                
                # Extract embeddings and compute score
                
                ## sample the sentence based on DPP
                sentences = []
                all_ot_scores = []
                for i in range(len(sampled_indices)):
                    n_shot_data = ([prompt_gen_data[0][j] for j in sampled_indices[i]], [prompt_gen_data[1][j] for j in sampled_indices[i]])
                    n_shot_text = d_template.fill(n_shot_data)
                    
                    # compute the linear kernel matrix for the sampled indices
                    features_examplars = all_examplars_representation[sampled_indices[i],:]
                    
                    # compute the similarity score to eval dataset using the optimal transport
                    distance_matrix = 1 - torch.mm(features_examplars, eval_examplars_representation.t()).cpu().numpy()
                    a, b = ot.unif(len(features_examplars)), ot.unif(len(eval_examplars_representation))  # uniform distribution on samples
                    ot_score = ot.emd(a, b, distance_matrix, log=True)[1]['cost']
                    all_ot_scores += [torch.tensor(ot_score)]

                    sentences.append(n_shot_text)
                # select the top determinants 
                top_indices = torch.stack(all_ot_scores).topk(batch_size, largest=False).indices
                sentences = [sentences[i] for i in top_indices]
                sampled_indices = sampled_indices[top_indices]
                                
                sentence_embeddings = model_forward_api.get_hidden_states(sentences)
                scores = -l.eval(sentence_embeddings)
                
                best = scores.min(dim=0)

                if best.values < top_score:
                    top_score = best.values
                    topk = sampled_indices[best.indices[0]]
                    
            print('sampling used {} seconds'.format(time.time() - start_time))

        with torch.no_grad():
            print('topk', topk)
            
            n_shot_data = ([prompt_gen_data[0][j] for j in topk], [prompt_gen_data[1][j] for j in topk])
            n_shot_text = d_template.fill(n_shot_data)
            new_sentence = n_shot_text
            
            new_sentence_embeddings = model_forward_api.get_hidden_states(new_sentence)
            score = -l.eval(new_sentence_embeddings)
            
            r, _, instruction = model_forward_api.eval_exemplars(prompts, topk, n_shot_data)
            # print([demos[j] for j in topk])
            print(n_shot_text)
            exemplars_log.append(n_shot_text)
            indices_log.append(','.join([str(z) for z in topk.clone().detach().cpu().tolist()]))
            
            print('Score:', score.item(), '; Actual perfromance:', -r)

        if r > best_r:
            best_r = r
            if r >= 1:
                break

        if t < STOP_TRAINING_AFTER_ITER:
            l.train(new_sentence_embeddings, r, NN_training_iter)
            
        instruction_log.append(instruction)
        r_log.append(r)
        
        print("iter {0} --- reward: {1}".format(t, r))
        print(f"Best value found till now: {best_r}")

    print('Evaluate on test data...')
    prompts = model_forward_api.return_best_prompt()
    print("Best instruction is:")
    print(prompts)

    prompts_set = model_forward_api.return_prompts_set()
    print("The final instruction set is:")
    print(model_forward_api.return_prompts_set())
    
    best_exemplars = model_forward_api.return_best_exemplars()
    print("The best exemplars are:")
    print(best_exemplars)
    
    best_exemplar_indices = model_forward_api.return_best_exemplar_indices()
    print("The best exemplar indices are:")
    print(best_exemplar_indices)
    
    
    try:
        if type(best_exemplars[0].item()) == int:
            best_exemplars = [d_template.fill([[prompt_gen_data[0][i]], [prompt_gen_data[1][i]]]) + '\n\n' for i in best_exemplars]
    except Exception as err:
        print(err)

    test_score, model_outputs, answers = None, None, None
    
    # Evaluate on test data
    print('Evaluating on test data...')

    test_conf = {
        'generation': {
            'num_subsamples': 3,
            'num_demos': 5,
            'num_prompts_per_subsample': 0,
            'model': {
                'gpt_config': {
                    'model': 'vicuna'
                }
            }
        },
        'evaluation': {
            'method': exec_accuracy_evaluator, # option: accuracy (cannot use likelihood here due to the textual outputs from ChatGPT do not have log prob)
            'task': task,
            'num_samples': min(100, len(test_data[0])),
            'model': {
                "name": "GPT_forward",
                'gpt_config': {
                   'model': gpt_model,
                }
            }
        }
    }
    
        
    conf = config.update_config(test_conf, base_conf)
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    
    best_n_shot_data = ([prompt_gen_data[0][i] for i in best_exemplar_indices], [prompt_gen_data[1][i] for i in best_exemplar_indices])
    
    if api_model == 'chatgpt':
        test_res, instruction_score, model_outputs, answers = conf['evaluation']['method'](prompts=prompts,
                                        eval_template=eval_template,
                                        eval_data=test_data,
                                        few_shot_data=prompt_gen_data,
                                        demos_template=demos_template,
                                        config=conf['evaluation'],
                                        given_demos=best_n_shot_data)
        # test_res = test_res[0]
        test_score = test_res.sorted()[1][0]
    else:
        test_conf = config.update_config(test_conf, base_conf)
        eval_template = template.EvalTemplate(eval_template)
        test_res, instruction_score, model_outputs, answers =  conf['evaluation']['method'](prompts, 
                                                                    eval_template, test_data, 
                                                                    d_template, prompt_gen_data, 
                                                                    test_conf['evaluation'], 
                                                                    model_api=model_forward_api, given_demos=best_n_shot_data)
        test_score = test_res.sorted()[1][0]
        print('Test score is', test_score)
    print(f'Test score on {api_model}: {test_score}')

    return test_score, prompts, prompts_set, best_exemplars, instruction_log, model_outputs, answers, r_log, exemplars_log, indices_log, best_r, best_init

def parse_args():
    parser = argparse.ArgumentParser(description="Induction pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--HF_cache_dir",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Your embedding model directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Set the seed."    
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.1,
        help="The weight decay value."
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=1,
        help="The standard deviation of noise in labels for NN training."
    )
    parser.add_argument(
        "--max_prompt_gen_size",
        type=int,
        default=100,
        help="The maximum number of examples pool  we use for prompot generation."
    )
    parser.add_argument(
        "--NN_training_iter",
        type=int,
        default=1000,
        help="Learning rate for the NN fitting."
    )
    parser.add_argument(
        "--n_domain",
        type=int,
        default=1000,
        help="Domain size for surrogate evaluation."
    )
    parser.add_argument(
        "--gpt_model",
        type=str,
        default='gpt-3.5-turbo-1106',
        help="gpt version to use."
    )
    parser.add_argument(
        "--num_shot",
        type=int,
        default=5,
        help="How many shots for in-context learning"    
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=0,
        help="Trial ID"    
    )
    parser.add_argument(
        "--expname",
        type=str,
        default='default',
        help="The name of the experiments."    
    )
    parser.add_argument(
        "--total_iter",
        type=int,
        default=165,
        help="The total number of iterations."    
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=40,
        help="The number of initial iterations."    
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(set_all_seed(args.seed))
    print(args)
    test_score, prompts, prompts_set, best_exemplars, instruction_log, model_outputs, answers, r_log, exemplars_log, indices_log, best_r, best_init = run(
        task=args.task,
        HF_cache_dir=args.HF_cache_dir,
        lam=args.lam,
        nu=args.nu,
        max_prompt_gen_size=args.max_prompt_gen_size,
        NN_training_iter=args.NN_training_iter,
        n_domain=args.n_domain,
        gpt_model=args.gpt_model,
        num_shot=args.num_shot,
        total_iter=args.total_iter,
        n_init=args.n_init
    )
    
    args_dict = vars(args)
    args_dict['api_model'] = api_model
    args_dict['test_score'] = test_score
    args_dict['valid_score'] = best_r
    args_dict['best_init'] = best_init
    args_dict['best_prompt'] = prompts
    args_dict['prompt_set'] = {tmp: prompts_set[tmp][0] for tmp in prompts_set}
    args_dict['best_exemplars'] = best_exemplars
    args_dict['exemplars_log'] = exemplars_log
    args_dict['indices_log'] = indices_log
    args_dict['instruction_log'] = instruction_log
    args_dict['r_log'] = r_log
    args_dict['model_outputs'] =  model_outputs
    args_dict['answers'] =  answers

    save_dir = "./results/systematic_experiments"
    save_dir = save_dir + '/' + args.expname
    # if the folder does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # get a path with the current time
    path = os.path.join(save_dir, args.task + '_ease_{}_trial{}_{}_shot_sampling'.format(args.gpt_model, args.trial, args.num_shot) + datetime.datetime.now().strftime("-%Y-%m-%d_%H-%M-%S")+".json")

    with open(path, 'x') as fp:
        json.dump(args_dict, fp, indent=4)
    
    print("Finished!!!")
    print(f'Test score on ChatGPT: {test_score}')


