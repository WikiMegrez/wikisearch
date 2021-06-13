import argparse
import ray

from wiki_search import Dataset, Engine


if __name__ == '__main__':
    if not ray.is_initialized():
        ray.init()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()

    dataset = Dataset(data_dir=args.data_dir)
    engine = Engine(dataset=dataset)

    while True:
        query = input('> ')
        results = engine.search(query)[:5]

        print('Only showing top 5 results:')
        for result in results:
            print('=============')
            print(f'{result.document.title} :: {result.score:.4f}')
            print(result.document.raw_main_desc)
            print()
