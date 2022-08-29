import re
import os
import shutil
import random
import argparse
from utils_pdtb3 import *

data_path = "PDTB_3.0/data" 

annotation_path = os.path.join(data_path, 'gold/')
text_path = os.path.join(data_path, 'raw/')
    
for dirname in os.listdir(annotation_path):
    print(dirname)
    annotation_dir = os.path.join(annotation_path, dirname)
    text_dir = os.path.join(text_path, dirname)
    for filename in os.listdir(annotation_dir):
        print(filename)
        with open(os.path.join(annotation_dir, filename), encoding='latin1') as f:
            annotation_data = f.readlines()
            print(annotation_data)
        with open(os.path.join(text_dir, filename), encoding='latin1') as f:
            text_data = f.read()
            print(text_data)
        break
    break


def process_line(line, text_data, target_relation):
    """Processes a single line of annotated example in PDTB 3.0."""
    args = line.split('|')
    relation_type = args[0]
    
    """
    @brucewlee
    original: only Implicit available
    modified: Explicit also available
    """
    assert ['Explicit','Implicit'].count(target_relation) == 1, f"{target_relation} is not available"
    if relation_type != target_relation:
        return None

    conn1 = args[7]
    conn1_sense1 = args[8]
    conn1_sense2 = args[9]
    conn2 = args[10]
    conn2_sense1 = args[11]
    conn2_sense2 = args[12]

    arg1_idx = args[14].split(';')
    arg2_idx = args[20].split(';')

    arg1_str = []

    # Arguments may be discontiguous spans.
    for pairs in arg1_idx:
        arg1_i, arg1_j = pairs.split('..')
        arg1 = text_data[int(arg1_i):int(arg1_j)+1]
        arg1_str.append(re.sub('\n', ' ', arg1))

    arg2_str = []
    for pairs in arg2_idx:
        if pairs == '':
            continue
        arg2_i, arg2_j = pairs.split('..')
        arg2 = text_data[int(arg2_i):int(arg2_j)+1]
        arg2_str.append(re.sub('\n', ' ', arg2))

    return (conn1, conn1_sense1, conn1_sense2,
            conn2, conn2_sense1, conn2_sense2,
            ' '.join(arg1_str), ' '.join(arg2_str),
            relation_type)


def process_file(annotation_data, text_data, dirname, filename, target_relation):
    """Processes a single file of annotated examples in PDTB 3.0."""

    lines_to_write = []
    for line in annotation_data:
        data_tuple = process_line(line, text_data, target_relation)
        if data_tuple:
            conn1, conn1_sense1, conn1_sense2, \
            conn2, conn2_sense1, conn2_sense2, \
            arg1_str, arg2_str, relation_type = data_tuple

            lines_to_write.append(tab_delimited([dirname, filename, relation_type,
                                                 arg1_str, arg2_str,
                                                 conn1, conn1_sense1, conn1_sense2,
                                                 conn2, conn2_sense1, conn2_sense2]))
    return lines_to_write



def pdtb3_process_raw_sections(data_path, write_path, discourse_type):
    """Processes raw PDTB 3.0 data from LDC and saves individual sections to file."""

    annotation_path = os.path.join(data_path, 'gold/')
    text_path = os.path.join(data_path, 'raw/')
    
    for dirname in os.listdir(annotation_path):
        lines_to_write = []
        lines_to_write.append(tab_delimited(['section', 'filename',
                                             'relation_type', 'arg1', 'arg2',
                                             'conn1', 'conn1_sense1', 'conn1_sense2',
                                             'conn2', 'conn2_sense1', 'conn2_sense2',
                                             'sentence']))

        annotation_dir = os.path.join(annotation_path, dirname)
        text_dir = os.path.join(text_path, dirname)
        for filename in os.listdir(annotation_dir):
            with open(os.path.join(annotation_dir, filename), encoding='latin1') as f:
                annotation_data = f.readlines()
            with open(os.path.join(text_dir, filename), encoding='latin1') as f:
                text_data = f.read()

            lines_to_write.extend(process_file(annotation_data, text_data,
                                               dirname, filename, discourse_type))

        with open(f'{write_path}/{dirname}.tsv', 'w') as f:
            f.writelines(lines_to_write)
            print(f'Wrote Section {dirname}')



def main():
    #python preprocess_pdtb3.py --data_dir PDTB_3.0/data --output_dir pdtb3_xval

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None, type=str, required=True,
                        help='Path to a directory containing raw and gold PDTB 3.0 files.\
                              Refer to README.md about obtaining this file.')
    parser.add_argument('--output_dir', default=None, type=str, required=True,
                        help='Path to output directory \
                              where the preprocessed dataset will be stored.')

    args = parser.parse_args()


    """
    @brucewlee
    original: if sections/ exist, don't run pdtb3_process_raw_sections
    modified: if sections/ exist, delete path and write again
    """
    sections_data_dir = os.path.join(args.data_dir, 'sections/')
    if os.path.exists(sections_data_dir):
        shutil.rmtree(sections_data_dir)
        print(f'Delete Old Directory {sections_data_dir}')
    os.makedirs(sections_data_dir)
    pdtb3_process_raw_sections(args.data_dir, sections_data_dir, args.discourse_type)

    # Create splits.
    if args.split == 'L2_xval':
        pdtb3_make_splits_xval(sections_data_dir, args.output_dir, level='L2')
    elif args.split == 'L3_xval':
        pdtb3_make_splits_xval(sections_data_dir, args.output_dir, level='L3')
    elif args.split == 'L1_ji':
        pdtb3_make_splits_l1(sections_data_dir, args.output_dir)
    else:
        raise ValueError('--split must be one of "L2_xval", "L3_xval", "L1_ji".')