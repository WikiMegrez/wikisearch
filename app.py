from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from wiki_search import Dataset, Engine

app = FastAPI()


app.mount("/webui", StaticFiles(directory="webui"), name="webui")


dataset = Dataset(data_dir='./data')
engine = Engine(dataset=dataset, ranking_algo='tfidf')


@app.get('/search/')
def search(q: str):
    results = engine.search(q)
    res = [x.document.main_image for x in results]

    def is_bullshit(x: str) -> bool:
        if x.endswith('20px-Semi-protection-shackle.svg.png'):
            return True
        if x.endswith('50px-Question_book-new.svg.png'):
            return True
        if x.endswith('40px-Edit-clear.svg.png'):
            return True
        if x.endswith('50px-Question_book-new.svg.png'):
            return True
        if x.endswith('19px-Symbol_support_vote.svg.png'):
            return True
        if x.endswith('40px-Text_document_with_red_question_mark.svg.png'):
            return True
        if x.endswith('40px-Ambox_important.svg.png'):
            return True
        return False

    res = [x for x in res if not is_bullshit(x)]
    return res
