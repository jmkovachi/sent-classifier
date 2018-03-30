import requests
import json
from classifiers.classifyPOS import Trainer as trainer
from datetime import datetime, timedelta
import re

def query_org_prices(org_name, dates):
    """
    Queries Quandl table for a prices object according to a certain org name over a range of dates.
    @param org_name: Name of organization/company to query
    @param dates: List of dates to query
    """

    url = "https://www.quandl.com/api/v3/datatables/WIKI/PRICES"

    # Below, we will join each date by a comma in order to query Quandl
    querystring = {"ticker":org_name,"date":','.join(dates),"api_key":"18EnimioewvUf_FuJxf-"}

    headers = {
        'cache-control': "no-cache",
        'postman-token': "391f1386-bac9-968b-85ce-4326796adec9"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    # load into json
    result = json.loads(response.text)

    print(result)

    # return opening prices from list of dates and closing prices from list of dates
    return {'open' : result['datatable']['data'][0][2], 'close' : result['datatable']['data'][0][5], 
            'final-open' : result['datatable']['data'][1][2], 'final-close' : result['datatable']['data'][1][5]}

def add_week(time):
    """
    Adds a week to a python time object.
    @param string: Date string to be converted
    return: converted time

    Example input and output is below.

    Input: 2009-01-05T07h53m00

    Return: 2009-01-12T07h53m00

    """
    t = datetime.strptime(time, '%Y-%m-%dT%Hh%Mm%S')
    t = t + timedelta(weeks=1)
    t = datetime.strftime(t, '%Y-%m-%dT%Hh%Mm%S')
    return t

def convert_dates(dates):
    """
    Converts dates from long-form format with hours and seconds to dates with just year, month, and day.

    Input: 2009-01-05T07h53m00

    Return: 2009-01-05
    """
    return [re.sub('T.*', '', date, flags=re.DOTALL) for date in dates]

class QuandlWrapper:

    def __init__(self):
        self.bayes_classifier = trainer()
        self.bayes_classifier.nltk_train_semeval()

    def classification_decision(self, title, org_name, date):
        decision = self.bayes_classifier.classify(text=title)
        price_data = query_org_prices(org_name, convert_dates([date, add_week(date)]))
        price_movement = price_data['final-close'] - price_data['open']
        if price_movement > 0 and decision == 'positive':
            return 'correct'
        elif price_movement > 0 and decision == 'negative':
            return 'false'
        elif price_movement < 0 and decision == 'positive':
            return 'false'
        elif price_movement < 0 and decision == 'negative':
            return 'true'


         




quandl = QuandlWrapper()

print(quandl.query_org_prices('PCG', ['2004-11-18','2004-11-23']))