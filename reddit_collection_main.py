import praw
from psaw import PushshiftAPI
import calendar
from reddit_scraper import corpus_loop

#Declare api
api = PushshiftAPI()

reddit = praw.Reddit(client_id='zcgbF4hEJmKQymABChaTkA',        
                               client_secret='_SgbEtm0e_pbFZTfHzxCdwtXqkZmHg',
                               user_agent="reddit_scraper_city_subreddits by u\happyDuck54")

WRITE_PATH = ''

#Get dates for starting and ending corpus collection
last_days = {(k, m): calendar.monthrange(k, m)[1] for k in range(2021,2023) for m in range(1, 13)}

months = list(last_days.keys())
months = [i for i in months if i[1] < 7 and i[0] > 2021]
month_range = {(k, m): calendar.monthrange(k, m)[1] for k, m in months}

start_end_list = [((key[0],key[1],1),(key[0],key[1],value)) for key, value in month_range.items()]

print(start_end_list)

#Loop through dates and collect corpora from Reddit
corpus_loop(reddit, start_end_list, 'Portland', WRITE_PATH)