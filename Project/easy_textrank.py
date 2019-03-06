# Wikipedia scraper
import bs4 as bs  
import urllib.request  
import re
import nltk
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.summarization import summarize
from gensim.summarization import keywords

url_topull = input('Enter the Wikipedia URL to pull - ')

scraped_data = urllib.request.urlopen(url_topull)  
article = scraped_data.read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:  
    article_text += p.text

print("Data pull done")

print("==================================SUMMARY===================================")
print (summarize(article_text,ratio=0.01))

print("==================================KEYWORDS===================================")
print (keywords(article_text,ratio=0.01))
