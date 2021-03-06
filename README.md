# PyTerrier_ANCE

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [ANCE]https://github.com/microsoft/ANCE/) dense passage retriever.

## Installation

This repostory can be installed using Pip.

    pip install --upgrade git+https://github.com/terrierteam/pyterrier_ance.git

You will need FAISS (cpu or gpu) installed:

On Colab:

    !pip install faiss-cpu 
    
On Anaconda:

    # CPU-only version
    $ conda install -c pytorch faiss-cpu

    # GPU(+CPU) version
    $ conda install -c pytorch faiss-gpu

For ANCE, the CPU version is sufficient.

## Indexing

You will need a pre-trained ANCE checkpoint. There are several available from the [ANCE repository](https://github.com/microsoft/ANCE/#results).

Then, indexing is as easy as instantiating the indexer, pointing at the (unzipped) checkpoint and the directory in which you wish to create an index

```python

dataset = pt.get_dataset("irds:vaswani")
import pyterrier_ance
indexer = pyterrier_ance.ANCEIndexer("/path/to/checkpoint", "/path/to/anceindex")
indexer.index(dataset.get_corpus_iter())

```

## Retrieval

You can instantiate the retrieval transformer, again by specifying the checkpoint location and the index location:

```python
retr = pyterrier_ance.ANCERetrieval("/path/to/checkpoint", "/path/to/anceindex")
```

Thereafter, you can use it in the normal PyTerrier way, for instance in an experiment:

```python
pt.Experiment(
    [retr], 
    dataset.get_topics(), 
    dataset.get_qrels(), 
    eval_metrics=["map"]
)
```


## Credits

- Craig Macdonald, University of Glasgow
- Nicola Tonellotto, University of Pisa