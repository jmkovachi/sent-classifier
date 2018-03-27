import requests
import json

class QuandlWrapper:

    @staticmethod
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



quandl = QuandlWrapper()

print(quandl.query_org_prices('PCG', ['2004-11-18','2004-11-23']))