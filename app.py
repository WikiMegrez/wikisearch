from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wiki_search import Dataset, Engine

app = FastAPI()


app.mount("/webui", StaticFiles(directory="webui"), name="webui")


dataset = Dataset(data_dir='./data')
engine = Engine(dataset=dataset, ranking_algo='tfidf')


class Query(BaseModel):
    query: str


@app.get('/search/')
def search(q: str):
    results = engine.search(q)
    return [x.document.main_image for x in results]
