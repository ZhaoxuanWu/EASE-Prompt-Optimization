import torch
import random
import numpy as np

TASKS=[
    'antonyms', 'cause_and_effect', 'common_concept', 'diff', 'first_word_letter',
    'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 
    'num_to_verbal', 'active_to_passive', 'singular_to_plural', 'rhymes',
    'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
    'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
    'translation_en-fr', 'word_in_context', 'auto_categorization', 'auto_debugging', 'ascii', 'cs_algorithms',
    'periodic_elements', 'word_sorting', 'word_unscrambling', 'odd_one_out', 'object_counting',
    'sum_modified', 'sum_5digit', 'sum_5digit_noisy', 'multiply_3digit_noisy', 'regression_w17_noisy', 'regression_w17_noisy_half', 'regression_w7_noisy_half', 'regression_w17_noisy_new',
    'sst2-first200', 'trec', 'subj', 'sst5', 'web_questions', 'mtop', 'nl2bash', 'break', 'mnli', 'geoquery', 'commonsense_qa', 'sst5_noisy',
    'antonyms_noisy', 'antonyms_20_noisy', 'sentence_similarity_noisy',
    'nl2bash_20_noisy', 'object_counting_20_noisy', 'orthography_starts_with_20_noisy', 'second_word_letter_20_noisy', 
    'sentence_similarity_20_noisy', 'synonyms_20_noisy', 'word_sorting_20_noisy', 'word_unscrambling_20_noisy',
    'object_counting_50_noisy', 'orthography_starts_with_50_noisy', 'second_word_letter_50_noisy', 'sentence_similarity_50_noisy', 'synonyms_50_noisy', 'word_sorting_50_noisy', 'word_unscrambling_50_noisy',
    'object_counting_10_noisy', 'object_counting_20_noisy', 'object_counting_30_noisy', 'object_counting_40_noisy', 'object_counting_50_noisy', 'object_counting_60_noisy', 'object_counting_70_noisy', 'object_counting_80_noisy', 'object_counting_90_noisy',
    'sentence_similarity_10_noisy', 'sentence_similarity_20_noisy', 'sentence_similarity_30_noisy', 'sentence_similarity_40_noisy', 'sentence_similarity_50_noisy', 'sentence_similarity_60_noisy', 'sentence_similarity_70_noisy', 'sentence_similarity_80_noisy', 'sentence_similarity_90_noisy',
    'linear', 'linear_1', 'linear_2', 'linear_3', 'linear_4',
    'linear_4_10_noisy', 'linear_4_20_noisy', 'linear_4_30_noisy', 'linear_4_40_noisy', 'linear_4_50_noisy', 'linear_4_60_noisy', 'linear_4_70_noisy', 'linear_4_80_noisy', 'linear_4_90_noisy',
    'pig_latin_encode', 'pig_latin_encode_10_noisy', 'pig_latin_encode_30_noisy', 'pig_latin_encode_50_noisy', 'pig_latin_encode_70_noisy', 'pig_latin_encode_90_noisy',
    'geometric_shapes', 'geometric_shapes_50_noisy', 'anachronisms', 'analogical_similarity', 'checkmate_in_one', 'linguistics_puzzles', 'two_digit_multiplication_plus_one',
    'protein_interacting_sites', 'protein_interacting_sites_10_noisy', 'protein_interacting_sites_30_noisy', 'protein_interacting_sites_50_noisy', 'protein_interacting_sites_70_noisy', 'protein_interacting_sites_90_noisy',
]


def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    return f"Set all the seeds to {seed} successfully!"