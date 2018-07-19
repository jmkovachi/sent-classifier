from elasticsearch import Elasticsearch
es = Elasticsearch()


class QueryES:
    
    def __init__(self, db=None):
        if db != None:
            self.orgs = [org for org in db.companies.find()]

    def search_db_for_orgs(self, search_str):
        for org in self.orgs:
            if org['title'] != '' and org['title'] in search_str:
                return org

    @staticmethod
    def query():
        query = {
            "size" : 1000,
            "query": {
                "match": {
                    "date" : "2007"
                }
            }
        }
        hits = es.search(index='articles', doc_type='article', body=query)
        return hits['hits']['hits']
