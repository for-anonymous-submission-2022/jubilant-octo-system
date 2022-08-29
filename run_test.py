
import io
import os
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (AutoConfig, 
                          AutoModelForSequenceClassification, 
                          AutoTokenizer, AdamW, 
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )

from dataset import GeneralDataset
from test import GeneralTest
from model import ClassificationHead, SequenceClassification
import utils

from transformers import T5Tokenizer
import csv

parser = argparse.ArgumentParser()

# parser.add_argument('--epochs',
#                     default=5,
#                     type=int,
#                     help='(int) Number of training epochs')

parser.add_argument('--batch_size',
                    default=8,
                    type=int,
                    help='Number of batches - depending on the max sequence length and GPU memory. For 512 sequence length batch of 10 works without cuda memory issues. For small sequence length can try batch of 32 or higher.')

parser.add_argument('--max_length',
                    default=None,
                    type=int,
                    help='(int) Pad or truncate text sequences to a specific length. If `None` it will use maximum sequence of word piece tokens allowed by model.' )

parser.add_argument('--model_name_or_path',
                    default="-large",
                    type=str,
                    help='(str) Name of transformers model - will use already pretrained model. Path of transformer model - will load your own model from local disk.')

parser.add_argument('--test_path',
                    default='./PDTB/implicit_pdtb3_xval/fold_1/test.tsv',
                    type=str,
                    help='(str) Valid tsv/csv path.')

parser.add_argument('--tokenizer',
                    default=None,
                    type=str,
                    help='Name of tokenizer')

parser.add_argument('--config',
                    default=None,
                    type=str,
                    help='Name of config')


# parser.add_argument('--learning_rate',
#                     default=5e-6,
#                     type=float,
#                     help='learning rate')

args = parser.parse_args()


# Set seed for reproducibility,
set_seed(2022)

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dicitonary of labels and their id - this will be used to convert.
# String labels to number ids.
labels_ids =\
 {'Temporal.Asynchronous': 0, 'Temporal.Synchronous': 1, 
 'Contingency.Cause': 2, 'Contingency.Cause+Belief': 3, 
 'Contingency.Condition': 4, 'Contingency.Purpose': 5, 
 'Comparison.Contrast': 6, 'Comparison.Concession': 7, 
 'Expansion.Conjunction': 8, 'Expansion.Instantiation': 9, 
 'Expansion.Equivalence': 10, 'Expansion.Level-of-detail': 11, 
 'Expansion.Manner': 12, 'Expansion.Substitution': 13,
 'None': 14
 }

# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

# Get model configuration.
print('Loading configuration...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=args.config, num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.tokenizer)

# Get the actual model. (best ckpt)
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path, config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`'%device)

# Create pytorch dataset.
test_dataset =\
 GeneralDataset(path=args.test_path, 
                    use_tokenizer=tokenizer, 
                    sentence_col = 'arg1',
                    sentence_col_pair = 'arg2',
                    labels_ids =labels_ids,
                    labels_col ='label1',
                    labels_col_pair ='label2',
                    max_sequence_len = args.max_length)
print('Created `test_dataset` with %d examples!'%len(test_dataset))


# Move pytorch dataset into dataloader.
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
print('Created `test_dataloader` with %d batches!'%len(test_dataloader))


# Get prediction form model on test data. 
print('test on batches...')
test_labels, test_labels_pair, test_predictions, test_loss = GeneralTest.test(model, test_dataloader, device, pair = True)

test_acc = accuracy_score(test_labels, test_predictions)

duplicate_test_acc = utils.list_match(labels = test_labels, predictions = test_predictions, labels_pair = test_labels_pair)/(len(test_labels))

# Print loss and accuracy values to see how training evolves.
print("  test_loss: %.5f - test_acc: %.5f - duplicate_test_acc: %.5f"%(test_loss, test_acc, duplicate_test_acc))
    
    
# Store the average loss after each epoch so we can plot them.
all_loss = {'test_loss':[]}
all_acc = {'test_acc':[], 'duplicate_test_acc':[]}

# Store the loss value for plotting the learning curve.
all_loss['test_loss'].append(test_loss)
all_acc['test_acc'].append(test_acc)
all_acc['duplicate_test_acc'].append(duplicate_test_acc)



# prediction to text label
label_reversed_dict = {v:k for k, v in labels_ids.items()}


# loss, acc, prediction result save
score_file_name = 'test_result/' + 'test_score.tsv'
pred_file_name = 'test_result/' + args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', '') + '_test_pred.tsv'

# write score
if os.path.isfile(score_file_name):
    with open(score_file_name, 'a', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow([args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', ''), args.test_path.split('/')[-2], test_loss, test_acc, duplicate_test_acc])
    
else:
    with open(score_file_name, 'w', newline='') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['model', 'fold', 'test_loss', 'test_acc',  'duplicate_test_acc'])
        w.writerow([args.model_name_or_path.split('/')[-1].replace('_checkpoint.pt', ''), args.test_path.split('/')[-2], test_loss, test_acc, duplicate_test_acc])

# write result
with open(pred_file_name, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t')
    w.writerow(['arg1', 'arg2', 'label', 'label_pair', 'prediction'])
    for arg1_sentence, arg2_sentence, label_reversed_dict[label], label_reversed_dict[label_pair], label_reversed_dict[prediction] in zip(test_dataset.texts,  test_dataset.texts_pair, test_dataset.labels, test_dataset.labels_pair, test_predictions):
        w.writerow([arg1_sentence, arg2_sentence, label_reversed_dict[label], label_reversed_dict[label_pair], label_reversed_dict [prediction]])
        
        
print(all_loss)
print(all_acc)