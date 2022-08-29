# nohup python test_runner_gpu0.py > nohup_test/tester

import os
import time


# CUDA_VISIBLE_DEVICES=0 python -u run_test.py --model_name_or_path "ckpt/roberta-large/8_2e-06_roberta-large_fold1_checkpoint.pt" --tokenizer "roberta-large" --config "roberta-large" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_1/test.tsv"



# robart_large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_test.py --model_name_or_path "ckpt/roberta-large/8_2e-06_roberta-large_fold_{idx}_checkpoint.pt" --tokenizer "roberta-large" --config "roberta-large" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_{idx}/test.tsv" > nohup_test/roberta_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    

# # # # albert-large-v1
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_test.py --model_name_or_path "ckpt/albert-large/8_5e-06_albert-large-v1_fold_{idx}_checkpoint.pt" --tokenizer "albert-large-v1" --config "albert-large-v1" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_{idx}/test.tsv" > nohup_test/albert-large-v1_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")

    
# # # # # bart_large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_test.py --model_name_or_path "ckpt/bart-large/8_5e-06_bart-large_fold_{idx}_checkpoint.pt" --tokenizer "facebook/bart-large" --config "facebook/bart-large" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_{idx}/test.tsv" > nohup_test/bart_large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")
    
    
# # # microsoft/deberta-large
# for idx in range(1,13):
#     start = time.time()
#     os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_test.py --model_name_or_path "ckpt/deberta-large/8_2e-06_deberta-large_fold_{idx}_checkpoint.pt" --tokenizer "microsoft/deberta-large" --config "microsoft/deberta-large" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_{idx}/test.tsv" > nohup_test/deberta-large_fold_{idx}')
#     end = time.time()
#     print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")


# spanbert
# "SpanBERT/spanbert-large-cased"
for idx in range(1,13):
    start = time.time()
    os.system(f'CUDA_VISIBLE_DEVICES=0 nohup python -u run_test.py --model_name_or_path "ckpt/spanbert-large/8_5e-06_spanbert-large-cased_fold_{idx}_checkpoint.pt" --tokenizer "SpanBERT/spanbert-large-cased" --config "SpanBERT/spanbert-large-cased" --batch_size 8 --max_length 256 --test_path "PDTB/implicit_pdtb3_xval/fold_{idx}/test.tsv" > nohup_test/spanbert-large_fold_{idx}')
    end = time.time()
    print(f"---------- Time taken for fold {idx}: {end - start:.5f} sec ----------")