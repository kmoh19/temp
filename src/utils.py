"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Utilities
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

datasets = {}
instruction_datasets = {}
train_size = 0.8
val_size = 0.1
test_size = 0.1


def split_dataset(df, train_size=train_size, val_size=val_size, test_size=test_size, stratify=False, verbose=False):
    """
    Split the dataset and add a "split" column
    """
    df.dropna(inplace=True)
    if stratify:
        # Stratified sampling with labels
        df_train, df_val_test = train_test_split(
            df, train_size=train_size, shuffle=True, stratify=df['label'], random_state=0)
        df_val, df_test = train_test_split(
            df_val_test, train_size=val_size / (test_size + val_size),
            shuffle=True, stratify=df_val_test['label'], random_state=0,
        )
    else:
        df_train, df_val_test = train_test_split(
            df, train_size=train_size, shuffle=True, random_state=0)
        df_val, df_test = train_test_split(
            df_val_test, train_size=val_size / (test_size + val_size), shuffle=True, random_state=0)

    df_train['split'] = 'train'
    df_val['split'] = 'val'
    df_test['split'] = 'test'
    df = pd.concat([df_train, df_val, df_test]).sort_index()

    if verbose:
        print('Split:')
        print(df.split.value_counts())

    return df


def assign_instructions(df, outputs=None, instructions=[None], verbose=False):
    """
    Map labels and assign intructions
    """
    # Get outputs
    if outputs is None:
        outputs = sorted(df.output.unique().tolist())

    # Map labels
    if outputs != False:
        output2label = {output: label for label, output in enumerate(outputs)}
        df['label'] = df['output'].apply(output2label.get)

    # Assign instructions
    np.random.seed(0)
    df['instruction'] = np.random.choice(
        instructions, size=len(df), replace=True)

    if verbose and 'label' in df:
        print('Label:')
        print(df.label.value_counts())

    return df


def drop_long_sequences(df, max_chars=2000):
    """
    Drop long sequences by the maximum number of characters
    """
    df['num_chars'] = df['input'].str.len()
    df_new = df[df['num_chars'] <= max_chars].copy()
    keep_ratio = len(df_new) / len(df)
    print(f'{df.iloc[0].dataset.upper()}: keep {len(df_new)} of {len(df)} ({keep_ratio:.2%}), '
          f'drop {len(df) - len(df_new)} ({1 - keep_ratio:.2%})')
    return df_new[df_new.columns.drop('num_chars')]
