import jieba
from GraphRetriever.rank_bm25 import BM25Okapi
from nltk.stem.snowball import SnowballStemmer


def BM25_retriever(corpus, query, topk=3, return_scores=True):


    tokenized_corpus =[jieba.lcut(doc) for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = jieba.lcut(query)
    topk_result, match_cell_id = bm25.get_top_n(tokenized_query, corpus, n=topk)

    if return_scores:
        scores = bm25.get_scores(tokenized_query)
        scores = [scores[i] for i in match_cell_id.tolist()]
        return  topk_result,match_cell_id.tolist(),scores

    return topk_result,match_cell_id.tolist()

def EM_retriever(corpus, query, isEM=False,topk=3):
    result = {}
    snowball_stemmer = SnowballStemmer("english")

    query_stem = ' '.join([snowball_stemmer.stem(item).lower() for item  in query.split(' ')])
    corpus_stem = []
    for item in corpus:
        temp = []
        for i in item.split(' '):
            temp.append(snowball_stemmer.stem(i).lower())
        corpus_stem.append(' '.join(temp))
    for i in range(len(corpus_stem)):
        cell = corpus_stem[i]
        if len(query_stem.split(' ')) == 1:
            if cell == query_stem:
                result[i] = corpus[i]
        elif len(cell.split(' ')) == 1:
            if cell in query_stem.split(' ') :
                result[i] = corpus[i]
        elif cell in query_stem or query_stem in cell :
                result[i] = corpus[i]
    return list(result.values()),list(result.keys()),None

