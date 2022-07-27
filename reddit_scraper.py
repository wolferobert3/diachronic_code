import praw
from psaw import PushshiftAPI
import datetime as dt
import pickle
from os import path

#Functions for collecting submissions and comments from Reddit using PushShift

api = PushshiftAPI()

#Collect both submissions and their comments given a date range
#Return dictionary for both submissions and for comments, with keys based on the submission
#Note: dates should be in Year-Month-Day Format
def collect_submissions_and_comments(reddit, begin_date, end_date, subreddit_):

  start_time = int(dt.datetime(begin_date[0],begin_date[1],begin_date[2]).timestamp())
  end_time = int(dt.datetime(end_date[0],end_date[1],end_date[2]).timestamp())

  date_submissions = list(api.search_submissions(after=start_time, before=end_time, subreddit=subreddit_))
  date_ids = [sub.id for sub in date_submissions]

  submission_dict, comment_dict = {}, {}

  for id in date_ids:
    submission = reddit.submission(id)
    submission_dict[id] = submission

    submission.comments.replace_more(limit=None)
    submission_all_comments = submission.comments.list()
    
    comment_dict[id] = submission_all_comments

  num_submissions = len(list(submission_dict.keys()))
  print(f'Obtained {num_submissions} submissions with accompanying comments from date range')

  print('Returning submission_dict, comment_dict')

  return submission_dict, comment_dict

#Collect only submissions, given a date range
def collect_submissions(reddit, begin_date, end_date, subreddit_):

  start_time = int(dt.datetime(begin_date[0],begin_date[1],begin_date[2]).timestamp())
  end_time = int(dt.datetime(end_date[0],end_date[1],end_date[2]).timestamp())

  date_submissions = list(api.search_submissions(after=start_time, before=end_time, subreddit=subreddit_))
  date_ids = [sub.id for sub in date_submissions]

  submission_dict, comment_dict = {}, {}

  for id in date_ids:
    submission = reddit.submission(id)
    submission_dict[id] = submission

  num_submissions = len(list(submission_dict.keys()))
  print(f'Obtained {num_submissions} submissions from date range')

  print('Returning submission_dict')

  return submission_dict

#Collect only comments, given a date range
def collect_comments(reddit, begin_date, end_date, subreddit_):

  start_time = int(dt.datetime(begin_date[0],begin_date[1],begin_date[2]).timestamp())
  end_time = int(dt.datetime(end_date[0],end_date[1],end_date[2]).timestamp())

  date_submissions = list(api.search_submissions(after=start_time, before=end_time, subreddit=subreddit_))
  date_ids = [sub.id for sub in date_submissions]

  comment_dict = {}

  for id in date_ids:
    submission = reddit.submission(id)

    submission.comments.replace_more(limit=None)
    submission_all_comments = submission.comments.list()
    
    comment_dict[id] = submission_all_comments

  num_submissions = len(list(date_ids))
  print(f'Obtained comments from {num_submissions} submissions in date range')

  print('Returning comment_dict')

  return comment_dict

#Write pickle file to directory
def write_pickle(path_, object_):

  with open(path_,'wb') as pkl_writer:
    pickle.dump(object_, pkl_writer)

  print(f'Wrote pickle to {path_}')

  return

#Read pickle file from path
def read_pickle(path_):

  with open(path_,'rb') as pkl_reader:
    object_ = pickle.load(pkl_reader)

  print(f'Read pickle from {path_}')

  return object_


#Corpus creation functions

#Create a corpus file from a submission dictionary and comment dictionary
def create_corpus_list_from_subs_and_comments(submission_dict, comment_dict):

  corpus_list = []

  for id, comment_list in comment_dict.items():

    #Note -- may be text, not body for submissions
    submission_text = str(submission_dict[id].selftext)
    if submission_text != '[deleted]':
      corpus_list.append(submission_text)

    for comment in comment_list:
      corpus_list.append(str(comment.body))

  return corpus_list

#Create a corpus file from a submission dictionary
def create_corpus_list_from_submissions(submission_dict):

  corpus_list = []

  for id in submission_dict.keys():

    submission_text = str(submission_dict[id].selftext)
    if submission_text != '[deleted]':
      corpus_list.append(submission_text)

  return corpus_list

#Create a corpus file from a comment dictionary
def create_corpus_list_from_comments(comment_dict):

  corpus_list = []

  for id, comment_list in comment_dict.items():

    for comment in comment_list:

      corpus_list.append(str(comment.body))

  return corpus_list

#Return corpus list as a string
def create_text_corpus_from_list(corpus_list):

  text_corpus = '\n'.join(corpus_list)

  return text_corpus

#Return merged corpora as a string; input is a tuple of text corpora
def merge_text_corpora(text_corpora):

  merged_corpus = ''

  for corpus in text_corpora:
    merged_corpus += corpus

  return merged_corpus
  
def write_corpus(corpus, path_):

  with open(path_, 'w') as writer:
    writer.write(corpus)
  
  print(f'Wrote corpus to {path_}.')

  return

def read_corpus(path_):

  with open(path_, 'r') as reader:
    corpus = reader.read()
  
  print(f'Read corpus from {path_}.')

  return corpus

#Loops over a series of dates, gets corpora between them, and writes to file

def corpus_loop(reddit, dates, subreddit_, write_path):

  for date_tuple in dates:

    start_date, end_date = date_tuple[0], date_tuple[1]
    print(start_date)
    print(end_date)
    submission_dict, comment_dict = collect_submissions_and_comments(reddit, start_date, end_date, subreddit_)
    print('dicts done')

    submission_dict_write_path = path.join(write_path, f'{subreddit_}_{start_date[0]}_{start_date[1]}_submissions.pkl')
    comment_dict_write_path = path.join(write_path, f'{subreddit_}_{start_date[0]}_{start_date[1]}_comments.pkl')

    write_pickle(submission_dict_write_path, submission_dict)
    write_pickle(comment_dict_write_path, comment_dict)
    print('dicts written')

    corpus_list = create_corpus_list_from_subs_and_comments(submission_dict,comment_dict)
    print('created list')
    text_corpus = create_text_corpus_from_list(corpus_list)
    print('created corpus')

    corpus_write_path = path.join(write_path, f'{subreddit_}_{start_date[0]}_{start_date[1]}_corpus.txt')
    write_corpus(text_corpus, corpus_write_path)
    print('wrote corpus')

    print(f'Completed corpora processing for {start_date[0]}-{start_date[1]}')

  return