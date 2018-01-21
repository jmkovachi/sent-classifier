"""
This is news.py
"""
import newspaper

nyt_paper = newspaper.build("http://www.wsj.com",memoize_articles=False, language='en')
print(len(nyt_paper.articles))
    
for i in nyt_paper.articles:
    i.download()
    i.parse()
    print(i.text)
#article = cnn_paper.articles[0]


