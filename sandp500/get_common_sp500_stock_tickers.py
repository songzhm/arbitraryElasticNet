import requests
from bs4 import BeautifulSoup as bs
import numpy as np

import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from datetime import date


def get_common_sp500_stock_tickers(start_date, end_date):

    website_url = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies').text

    soup = bs(website_url, 'lxml')

    sp500_table = soup.find('table', {'class': 'wikitable sortable', 'id': 'constituents'})

    symbols_reports = sp500_table.findAll('a', {'class': 'external text'})

    symbols = []
    for i,s in enumerate(symbols_reports):
        if i % 2 ==0:
            symbols.append(s.contents[0])
    symbols = np.array(symbols, dtype=str)

    change_table = soup.findAll('table', {'class': 'wikitable sortable'})[1]

    changed_itmes = change_table.findAll('tr')[2:]

    dates = []
    added = []
    removed = []


    for i, item in enumerate(changed_itmes):
        
        all_tds = item.findAll('td')
        date_ = all_tds[0].contents[0].rstrip()
        try:
            dates.append(datetime.datetime.strptime(date_, "%B %d, %Y").date())
    #         d = datetime.datetime.strptime(date_, "%B %d, %Y")
    #         dates.append(date_)
            if(len(all_tds[1].contents)>0):
                added.append(all_tds[1].contents[0])
            else:
                added.append('')
            if(len(all_tds[3].contents)>0):
                removed.append(all_tds[3].contents[0])
            else:
                removed.append('')
        except:
            last_date = dates[-1]
            dates.append(last_date)
            if(len(all_tds[0].contents)>0):
                added.append(all_tds[0].contents[0])
            else:
                added.append('')
            if(len(all_tds[2].contents)>0):
                removed.append(all_tds[2].contents[0])
            else:
                removed.append('')

    changes = pd.DataFrame({'date': dates, 'added': added, 'removed': removed})

    # end_date = date.today()
    # start_date = end_date - relativedelta(years=n_year)

    valid_changes = changes[((changes.date>=start_date) & (changes.date<=end_date))]    

    changed_tickers = np.array(valid_changes.added.tolist() + valid_changes.removed.tolist())

    changed_tickers = np.unique(changed_tickers)

    valid_symbols = symbols[~np.isin(symbols, changed_tickers)]

    return list(np.unique(valid_symbols.tolist()))


# if __name__=='__main__':
#     print(get_common_sp500_stock_tickers(3))