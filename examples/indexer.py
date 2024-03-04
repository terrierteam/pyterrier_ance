import pyterrier as pt
import os
import pyterrier_ance


pt.init()
dataset = pt.datasets.get_dataset("msmarco_passage")


BASE_PATH = os.getcwd()
print(os.path.join(BASE_PATH, "content/anceindex"))

indexer = pyterrier_ance.ANCEIndexer(checkpoint_path="content/Passage ANCE(FirstP) Checkpoint",
                                     index_path="content/anceindex",
                                     num_docs=8841823,
                                     device='cuda:0')
indexer.index(dataset.get_corpus_iter())