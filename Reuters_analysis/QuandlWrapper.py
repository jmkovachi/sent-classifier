import requests
import json
#from classifyPOS import NB_Trainer as trainer
from datetime import datetime, timedelta
import re



def add_week(time):
    """
    Adds a week to a python time object.
    @param string: Date string to be converted
    return: converted time

    Example input and output is below.

    Input: 2009-01-05T07h53m00

    Return: 2009-01-12T07h53m00

    """
    t = datetime.strptime(time, '%Y-%m-%dT%Hh%Mm%S' )
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

def query_org_prices(org_name, dates):
    """
    Queries Quandl table for a prices object according to a certain org name over a range of dates.
    @param org_name: Name of organization/company to query
    @param dates: List of dates to query
    """

    print(dates)
    url = "https://www.quandl.com/api/v3/datatables/WIKI/PRICES"

    # Below, we will join each date by a comma in order to query Quandl
    querystring = {"ticker":org_name[0],"date":','.join(dates),"api_key":"18EnimioewvUf_FuJxf-"}

    headers = {
        'cache-control': "no-cache",
        'postman-token': "391f1386-bac9-968b-85ce-4326796adec9"
    }

    response = requests.request("GET", url, headers=headers, params=querystring)

    # load into json
    result = json.loads(response.text)
    # print(result)
    # return opening prices from list of dates and closing prices from list of dates
    return {'open' : result['datatable']['data'][0][2], 'close' : result['datatable']['data'][0][5], 
            'final-open' : result['datatable']['data'][1][2], 'final-close' : result['datatable']['data'][1][5]}

class QuandlWrapper:

    def __init__(self):
        self.bayes_classifier = None
        #self.bayes_classifier.nltk_train_semeval()
    
    def classification_decision(self, title, org_name, date, num_pos, num_neg):
        decision = self.bayes_classifier.classify(text=title)
        price_data = query_org_prices(org_name, convert_dates([date, add_week(date)]))
        price_movement = price_data['close'] - price_data['open']
        results = []
        if price_movement > 0 and decision == 'positive':
            results.append(True)
        elif price_movement > 0 and decision == 'negative':
            results.append(False)
        elif price_movement < 0 and decision == 'positive':
            results.append(False)
        elif price_movement < 0 and decision == 'negative':
            results.append(True)

        if price_movement > 0 and num_pos > num_neg:
            results.append(True)
        elif price_movement > 0 and num_neg > num_pos:
            results.append(False)
        elif price_movement < 0 and num_neg < num_pos:
            results.append(True)
        elif price_movement < 0 and num_pos > num_neg:
            results.append(False)
        elif num_pos == num_neg:
            results.append(True)

        return results
         




#quandl = QuandlWrapper()

#print(quandl.query_org_prices('PCG', ['2004-11-18','2004-11-23']))