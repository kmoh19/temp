"""
SPDX-License-Identifier: Apache-2.0
Copyright : JP Morgan Chase & Co

Data Preparation for CyberBench
"""

import os
import json
import pandas as pd

from ner_data import download_cyner, download_aptner, get_df_cyner, get_df_aptner
from sum_data import download_cynews, get_df_cynews
from mc_data import download_secmmlu, download_cyquiz, get_df_secmmlu, get_df_cyquiz
from tc_data import download_mitre, download_cve, download_web, download_email, download_http, \
    get_df_mitre, get_df_cve, get_df_web, get_df_email, get_df_http


if __name__ == "__main__":
    # Download datasets
    print('Downloading CyNER ...')
    download_cyner()
    print('Downloading APTNER ...')
    download_aptner()
    print('Downloading CyNews ...')
    download_cynews()
    print('Downloading SecMMLU ...')
    download_secmmlu()
    print('Downloading CyQuiz ...')
    download_cyquiz()
    print('Downloading MITRE ...')
    download_mitre()
    print('Downloading CVE ...')
    download_cve()
    print('Downloading Web ...')
    download_web()
    print('Downloading Email ...')
    download_email()
    print('Downloading HTTP ...')
    download_http()
    print('All downloading done!')

    # Collect datasets
    print('Loading CyNER ...')
    df_cyner = get_df_cyner()
    print('Loading APTNER ...')
    df_aptner = get_df_aptner()
    print('Loading CyNews ...')
    df_cynews = get_df_cynews()
    print('Loading SecMMLU ...')
    df_secmmlu = get_df_secmmlu()
    print('Loading CyQuiz ...')
    df_cyquiz = get_df_cyquiz()
    print('Loading MITRE ...')
    df_mitre = get_df_mitre()
    print('Loading CVE ...')
    df_cve = get_df_cve()
    print('Loading Web ...')
    df_web = get_df_web()
    print('Loading Email ...')
    df_email = get_df_email()
    print('Loading HTTP ...')
    df_http = get_df_http()
    print('All loading done!')

    # Save the CSV file (for generative models)
    dfs = [df_cyner, df_aptner, df_cynews, df_secmmlu,
           df_cyquiz, df_mitre, df_cve, df_web, df_email, df_http]
    columns = ['task', 'dataset', 'instruction', 'input', 'output', 'split']
    df_all = pd.concat([df[columns] for df in dfs], ignore_index=True)
    csv_file_path = os.path.join('data', 'cyberbench.csv')
    df_all.to_csv(csv_file_path, index=False)
    print(f'CyberBench data saved to {csv_file_path}')

    # Save the JSON file (for BERT-based models)
    datasets = {df.dataset.iloc[0]: df.to_dict(orient='records') for df in dfs}
    json_file_path = os.path.join('data', 'cyberbench.json')
    with open(json_file_path, 'w') as file:
        json.dump(datasets, file)
        print(f'CyberBench data Saved to {json_file_path}')

    # View datasets
    df_count = df_all.value_counts(subset=['task', 'dataset', 'split'])\
        .unstack()[['train', 'val', 'test']]
    df_count['sum'] = df_count.sum(axis='columns')
    print('-' * 50)
    print(df_count)
    print('-' * 50)
