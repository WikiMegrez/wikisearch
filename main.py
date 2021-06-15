import argparse
import ray

from wiki_search import Dataset, Engine


if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init()

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--ranking', type=str, default='tfidf')
    args = parser.parse_args()

    dataset = Dataset(data_dir=args.data_dir)
    engine = Engine(dataset=dataset, ranking_algo=args.ranking)

    while True:
        query = input('> ')
        # results = engine.search(query)[:5]
        results = engine.search(query)[:8]

        print('Only showing top 8 results:')
        for result in results:
            print('=============')
            print(f'{result.document.title} :: {result.score:.4f}')
            print('https://en.wikipedia.org/wiki/' + result.document.name)
            # print(result.document.main_image)
            desc = result.document.raw_main_desc
            if len(desc) > 1000:
                desc = desc[:1000]
                desc += '...'
            print(desc)
            print()
