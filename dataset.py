import os

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from ftfy import fix_text
import pandas as pd

from utils import multiple_replace


def select_sep(path: str):
    # Get file from path.
    if path[-3:] == "tsv":
        sep = "\t"
    elif path[-3:] == "csv":
        sep = ","
    elif path[-4:] == "json":
        sep = False
    return sep


class GeneralDataset(Dataset):
    """PyTorch Dataset class for loading data.

    This is where the data parsing happens and where the text gets encoded using
    loaded tokenizer.

    This class is built with reusability in mind: it can be used as is as long
        as the `dataloader` outputs a batch in dictionary format that can be passed 
        straight into the model - `model(**batch)`.

    Arguments:

        path (:obj:`str`):
            Path to the data partition.
        
        use_tokenizer (:obj:`transformers.tokenization_?`):
            Transformer type tokenizer used to process raw text into numbers.

        labels_ids (:obj:`dict`):
            Dictionary to encode any labels names into numbers. Keys map to 
            labels names and Values map to number associated to those labels.

        max_sequence_len (:obj:`int`, `optional`)
            Value to indicate the maximum desired sequence to truncate or pad text
            sequences. If no value is passed it will used maximum sequence size
            supported by the tokenizer and model.

    """

    def __init__(self, path: str, use_tokenizer: object, sentence_col: str, labels_ids: dict, labels_col: str, sentence_col_pair: str = None, labels_col_pair: str = None, max_sequence_len: int = None):
        # Check path.
        if not os.path.isfile(path):
            raise ValueError('Invalid `path` variable! Needs to be a file')

        # Check max sequence length. If not defined, use default.
        max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.texts = []
        self.texts_pair = []
        self.labels = []
        self.labels_pair = []
        
        # Read file
        sep = select_sep(path)
        if sep != False:
            data = pd.read_csv(path, sep=sep, on_bad_lines='warn')
        elif sep == False:
            data = pd.read_json(path, orient='records')
        else:
            print(f"data format at {path} is not supported")

        # Go through content.
        print(f'Reading {path}...')
        data_dict = data.to_dict('records')
        for row in tqdm(data_dict):
            content = fix_text(row[sentence_col]) # Fix any unicode issues.
            self.texts.append(content) # Save content.
            if sentence_col_pair != None:
                content_pair = fix_text(row[sentence_col_pair])
                self.texts_pair.append(content_pair) 

            label_id = labels_ids[row[labels_col]]
            self.labels.append(label_id) # Save encode labels.
            if labels_col_pair != None:
                if row[labels_col_pair] == None:
                    label_id_pair = labels_ids['None']
                else:
                    label_id_pair = labels_ids[row[labels_col_pair]]
                self.labels_pair.append(label_id_pair)

        # Number of instances.
        self.n_instances = len(self.labels)

        # Use tokenizer on texts. This can take a while.
        print('Using tokenizer on all texts. This can take a while...')
        if sentence_col_pair != None:
            self.inputs = use_tokenizer(self.texts, self.texts_pair, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt', max_length=max_sequence_len)
        else:
            self.inputs = use_tokenizer(self.texts, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt', max_length=max_sequence_len)

        # Get maximum sequence length.
        print('Texts padded or truncated to %d length!' % max_sequence_len)

        # Add labels.
        self.inputs.update({'labels':torch.tensor(self.labels)})
        if labels_col_pair != None:
            self.inputs.update({'labels_pair':torch.tensor(self.labels_pair)})
        print('Finished!\n')


    def __len__(self):
        """When used `len` return the number of instances.

        """
        return self.n_instances


    def __getitem__(self, item):
        """Given an index return an example from the position.
        
        Arguments:

        item (:obj:`int`):
            Index position to pick an example to return.

        Returns:
        :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
        It holddes the statement `model(**Returned Dictionary)`.

        """
        return {key: self.inputs[key][item] for key in self.inputs.keys()}