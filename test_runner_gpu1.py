# nohup python test_runner_gpu1.py > nohup_test/tester

import os
import time

# CUDA_VISIBLE_DEVICES=0 python -u run_test.py --model_name_or_path "ckpt/roberta-large/8_2e-06_roberta-large_fold1_checkpoint.pt" --tokenizer "roberta-large" --config "roberta-large" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_1/test.tsv"

# google/bigbird-roberta-large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_test.py --model_name_or_path "ckpt/bigbird-roberta-large/8_5e-06_bigbird-roberta-large_fold_{idx}_checkpoint.pt" --tokenizer "google/bigbird-roberta-large" --config "google/bigbird-roberta-large" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_{idx}/test.tsv" > nohup_test/bigbird_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

    
# # allenai/longformer-large-4096
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_test.py --model_name_or_path "ckpt/longformer/8_5e-06_longformer-large-4096_fold_{idx}_checkpoint.pt" --tokenizer "allenai/longformer-large-4096" --config "allenai/longformer-large-4096" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_{idx}/test.tsv" > nohup_test/longformer_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
