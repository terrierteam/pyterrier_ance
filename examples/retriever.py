import pyterrier as pt
import os, logging, sys, argparse, time, tqdm
import pyterrier_ance
import pandas as pd
import numpy as np

if not pt.started():
    pt.init()
    
    
def get_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format='[%(levelname)s] (%(asctime)s): %(message)s',
            level=logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    logger.info('Started')
    return logger


parser = argparse.ArgumentParser()
parser.add_argument('--topics', type=str, default='')
parser.add_argument('--name', type=str)
parser.add_argument('--path', type=str, default='result/')
parser.add_argument('--hits', type=int, default=10)
# parser.add_argument('--out', type=str, default='')

args = parser.parse_args()

'''
python example/retriever.py \
    --topics train.queries.tsv \
    --name train.queries \
    --path train.queries/ \
    --hits 1000
'''

def main(logger):
    start = time.time()
    ance_retr = pyterrier_ance.ANCERetrieval(checkpoint_path="content/Passage ANCE(FirstP) Checkpoint",
                                            index_path="content/anceindex",
                                            num_results=args.hits)


    train_queries = pd.read_csv(args.topics, names=['qid', 'query'], sep='\t', dtype=str, engine='pyarrow')
    os.makedirs('result/', exist_ok=True)
    split_number = (len(train_queries) // 500_000) + 1

    train_queries = np.array_split(train_queries, split_number)
    batch_on = 0
    for train_query in tqdm.tqdm(train_queries):
        res = ance_retr.transform(train_query)
        if 'query' in res.columns:
            res.drop(['query'], inplace=True, axis=1)
        res['tag'] = 1
        res[['qid', 'docid', 'docno', 'rank', 'score', 'tag']].to_csv(f'{args.path}ance.results.{args.name}.{batch_on}.tsv', sep='\t', index=False, header=False)
        batch_on += 1


    logger.critical('Finished ANCE retrieving, elapsed: {} min'.format( (time.time()-start) / 60))

if __name__ == '__main__':
    logger = get_logger()
    main(logger=logger)