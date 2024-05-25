import torch
import numpy as np
from torch.quasirandom import SobolEngine

default_dtype = torch.float32
EPSILON = np.finfo(np.float32).tiny

def gumbel_keys(w):
    # sample some gumbels
    max_val = 1.0
    uniform = torch.rand(w.shape).to(dtype=default_dtype) * (max_val - EPSILON) + EPSILON
    z = -torch.log(-torch.log(uniform))
    w = w + z.to(w.device)
    # print(z)
    # print(w)
    return w

def continuous_topk(w, k, t, separate=True):
    khot_list = []
    onehot_approx = torch.zeros_like(w, dtype=default_dtype)
    for i in range(k):
        khot_mask = torch.maximum(1.0 - onehot_approx, torch.tensor(EPSILON))
        w += torch.log(khot_mask)
        onehot_approx = torch.nn.functional.softmax(w / t, dim=-1)
        khot_list.append(onehot_approx)
    khot_list = torch.stack(khot_list)
    if separate:
        return khot_list
    else:
        return torch.sum(khot_list, 0)
    
def sample_subset(w, k, t=0.1, separate=True):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
        t (float): temperature of the softmax
    '''
    w = gumbel_keys(w)
    return continuous_topk(w, k, t, separate)

def remove_prefix(input_ids, outputs):
    """
    Args:
        input_ids (Tensor): batch_size * token_length
        outputs (Tensor): batch_size * output_token_length
    """
    assert input_ids.shape[0] == outputs.shape[0]

    prompt_length = input_ids.shape[1]
    return outputs[:, prompt_length:]

def draw_random_init(total_number, dimension):
    if dimension >= 21201:
        batch_size = 20000
        init_X = []
        n_batchs = dimension // batch_size + int((dimension % batch_size) != 0)
        for i in range(n_batchs):
            if i == n_batchs - 1:
                init_X.append(SobolEngine(dimension=dimension-batch_size*i, scramble=True, seed=i).draw(total_number))
            else:
                init_X.append(SobolEngine(dimension=batch_size, scramble=True, seed=i).draw(total_number))
        init_X = torch.cat(init_X, dim=1)
    else:
        init_X = SobolEngine(dimension=dimension, scramble=True, seed=0).draw(total_number)
    return init_X


def normalize_demos_tokens(tokens, attention):
    assert tokens.shape == attention.shape
    start_idx = torch.where(tokens == 1)[0]
    # Removing the bos token [1], the replace the token for "Input" with the variant without "\n"
    tokens = torch.cat([tokens[:start_idx], torch.tensor([4290], device=tokens.device), tokens[start_idx+2:]])
    attention = torch.cat([attention[:start_idx], torch.tensor([1], device=attention.device), attention[start_idx+2:]])
    return tokens, attention

def unnormalize_first_demos_tokens(tokens, attention):
    assert tokens.shape == attention.shape
    start_idx = torch.where(tokens == 4290)[0][0]
    tokens = torch.cat([tokens[:start_idx], torch.tensor([1, 10567], device=tokens.device), tokens[start_idx+1:]])
    attention = torch.cat([attention[:start_idx], torch.tensor([1, 1], device=attention.device), attention[start_idx+1:]])
    return tokens, attention

def normalize_instruction_tokens(tokens, attention):
    assert tokens.shape == attention.shape
    start_idx = torch.where(tokens == 1)[0]
    # Removing the bos token [1], the replace the token for "The" with the variant without "\n"
    tokens = torch.cat([tokens[:start_idx], torch.tensor([1576], device=tokens.device), tokens[start_idx+2:]])
    attention = torch.cat([attention[:start_idx], torch.tensor([1], device=attention.device), attention[start_idx+2:]])
    return tokens, attention


TASK_TO_BS = {'active_to_passive': 64, 'antonyms': 64,  'auto_categorization': 64, 'auto_debugging': 32, 'cause_and_effect': 32, 'common_concept': 64,
              'diff': 128,}

default_bs = 32
