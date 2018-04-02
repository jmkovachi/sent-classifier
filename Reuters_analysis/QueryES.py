from elasticsearch import Elasticsearch
es = Elasticsearch()


class QueryES:
    
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


#print(QueryES.query())