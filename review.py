import requests
from bs4 import BeautifulSoup
import pandas as pd

letter = []

class Sentiment:
    def get_soup(url):
        r = requests.get('https://mail.google.com/mail/u/0/#inbox')
        soup = BeautifulSoup(r.text, 'html.parser')
        return soup


    def get_reviews(soup):
        reviews = soup.find_all('div', {'data-hook': 'review'})
        try:
            for item in reviews:
                review = {
                'Category': soup.title.text.replace('https://mail.google.com/mail:title:', '').strip(),
                'Message': item.find('a', {'data-hook': 'message'}).text.strip(),
                               }
                letter.append(review)
        except:
            pass

    for x in range(10):
        soup = get_soup(f'https://mail.google.com/mail/u/0/#inbox={x}')
        print(f'Getting page: {x}')
        get_reviews(soup)
        print(len(letter))
        if not soup.find('li', {'class': 'a-disabled a-last'}):
            pass
        else:
            break

    df = pd.DataFrame(letter)
    print(df.head())
    df.to_csv(r'./data.csv', index=None)

