import json
from datetime import datetime
import pickle
import fasttext as ft
from os import listdir, path

def write_pickle(obj, path_):

    with open(path_,'wb') as pkl_writer:
        pickle.dump(obj, pkl_writer)

    return

def create_count_dict(json_list):

    count_dict = {}

    for json_str in json_list:

        result = json.loads(json_str)
        dt = datetime.fromtimestamp(result['timestamp'])
        str_dt = dt.strftime("%m/%Y")

        if str_dt in count_dict:
            count_dict[str_dt] += 1
        
        else:
            count_dict[str_dt] = 1

    return count_dict

def create_count_list(count_dict, years, months):

    count_list, date_list = [], []

    for year in years:
        
        for month in months:

            if month < 10:

                if f'0{month}/{year}' in count_dict:

                    count_list.append(count_dict[f'0{month}/{year}'])
                    date_list.append(f'0{month}/{year}')
                
                else:
                    
                    if f'{month}/{year}' in count_dict:
                        count_list.append(count_dict[f'{month}/{year}'])
                        date_list.append(f'{month}/{year}')

    return count_list, date_list

def create_utterance_dict(json_list, year = True, month = True, day = False):

    utterance_dict = {}

    str_time = ''

    if month:
        str_time += '%m'

    if day:
        str_time += '%d'

    if year:
        str_time += '%Y'


    for json_str in json_list:

        result = json.loads(json_str)
        dt = datetime.fromtimestamp(result['timestamp'])
        str_dt = dt.strftime(str_time)

        if str_dt in utterance_dict:
            utterance_dict[str_dt].append(result['text'])
        
        else:
            utterance_dict[str_dt] = [result['text']]

    return utterance_dict


def get_utterance_counts(utterance_dict):

    dates = list(utterance_dict.keys())

    utterances = []

    for date in dates:

        utterances.append(len(utterance_dict[date]))
    
    return dates, utterances

def get_token_counts(utterance_dict, month_year_list):

    token_list, date_list = [], []
    token_dict = {}

    for month_year in month_year_list:

        if month_year in utterance_dict:

            month_corpus = ' '.join(utterance_dict[month_year])
            month_tokens = month_corpus.split(' ')

            #Note: Could use a tokenization library like NLTK

            token_list.append(len(month_tokens))
            date_list.append(month_year)
            token_dict[month_year] = len(month_tokens)

    return token_dict, (date_list, token_list)


def create_year_month_list(start_year, start_month, end_year, end_month):

    month_year_list = []
    
    for year in range(start_year, end_year + 1):

        first_month, last_month = start_month, end_month
        
        if year != start_year:

            first_month = 1

        if year != end_year:

            last_month = 13

        for month in range(first_month, last_month):

            if month < 10:

                month_year_list.append(f'0{month}/{year}')

            else:

                month_year_list.append(f'{month}/{year}')

    return month_year_list

def create_text_corpora(utterance_dict, month_year_list, corpus_path, city = ''):

    for month_year in month_year_list:

        if month_year in utterance_dict:

            month_year_write_string = month_year.replace('/', '_')

            corpus = '\n'.join(utterance_dict[month_year])
            corpus_path_ = path.join(corpus_path, f'{city}_{month_year_write_string}.txt')

            with open(corpus_path_, 'w') as writer:
                writer.write(corpus)

    return

def train_ft_embeddings(corpus_path, vector_path):

    target_corpora = listdir(corpus_path)

    for corpus in target_corpora:

        ft_write_string = corpus[:-4]
        ft_target = path.join(corpus_path, corpus)

        ft_model = ft.train_unsupervised(corpus_path)

        ft_path = path.join(vector_path, f'{ft_write_string}.bin')
        ft_model.save_model(ft_path)

    return

def create_text_corpora_and_train_ft_embeddings(utterance_dict, month_year_list, corpus_path, vector_path, city = ''):

    for month_year in month_year_list:

        if month_year in utterance_dict:

            month_year_write_string = month_year.replace('/', '_')

            corpus = '\n'.join(utterance_dict[month_year])
            corpus_path_ = path.join(corpus_path, f'{city}_{month_year_write_string}.txt')

            with open(corpus_path_, 'w') as writer:
                writer.write(corpus)

            ft_model = ft.train_unsupervised(corpus_path)

            ft_path = path.join(vector_path, f'{city}_{month_year_write_string}.bin')
            ft_model.save_model(ft_path)

    return

#Preprocessing for FT?