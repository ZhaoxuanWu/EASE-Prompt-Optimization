export CUDA_VISIBLE_DEVICES="0"
source ./key

datasets=(ag_news_textlabel_redirect1_10_noisy)

for i in ${datasets[@]}; do
    for trial in 0 1 2; do # The number of trials
        # EASE
        python experiments/run_no_instruction_ease.py \
        --task $i \
        --lam 0.1 \
        --nu 0.01 \
        --num_shot 5 \
        --gpt_model gpt-3.5-turbo-1106 \
        --trial $trial \
        --expname exp_folder_name \
        --total_iter 165 \
        --n_init 40 \
        --seed 0

        # # Best-of-N
        # echo $i
        # python experiments/run_no_instruction_best_of_n.py \
        # --task $i \
        # --num_shot 5 \
        # --gpt_model gpt-3.5-turbo-1106 \
        # --expname exp_folder_name \
        # --n_init 165 \
        # --total_iter 165 \
        # --seed 0 \
        # --trial $trial 

        # # Evo
        # python experiments/run_no_instruction_sampling_evo.py \
        # --task $i \
        # --num_shot 5 \
        # --gpt_model gpt-3.5-turbo-1106 \
        # --expname exp_folder_name \
        # --total_iter 165 \
        # --n_init 40 \
        # --seed 0 \
        # --trial $trial
        
        # # Subset (DDP, MMD, OT)
        # python experiments/run_no_instruction_subset.py \
        # --task $i \
        # --num_shot 5 \
        # --gpt_model gpt-3.5-turbo-1106 \
        # --expname exp_folder_name \
        # --total_iter 165 \
        # --seed 0 \
        # --trial $trial \
        # --subset_method ot

        # # Retrieval (BM25, Cosine)
        # python experiments/run_no_instruction_retrieval.py \
        # --task $i \
        # --num_shot 5 \
        # --gpt_model gpt-3.5-turbo-1106 \
        # --expname exp_folder_name \
        # --total_iter 165 \
        # --seed 0 \
        # --trial $trial \
        # --retrieval_method bm25

        # # Inf
        # python experiments/run_no_instruction_inf.py \
        # --task $i \
        # --num_shot 5 \
        # --gpt_model gpt-3.5-turbo-1106 \
        # --expname exp_folder_name \
        # --total_iter 165 \
        # --seed 0 \
        # --trial $trial 

        # # Joint optimization for EASE
        # python experiments/run_choose_instruction_ease.py \
        # --task $i \
        # --lam 0.1 \
        # --nu 0.01 \
        # --num_shot 5 \
        # --gpt_model gpt-3.5-turbo-1106 \
        # --expname exp_folder_name \
        # --n_init 40 \
        # --total_iter 165 \
        # --seed 0 \
        # --trial $trial
    done
done
