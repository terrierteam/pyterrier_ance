# PyTerrier_ANCE

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [ANCE](https://github.com/microsoft/ANCE/) dense passage retriever.

## Installation

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
anceretr = pyterrier_ance.ANCERetrieval("/path/to/checkpoint", "/path/to/anceindex")
```

Thereafter, you can use it in the normal PyTerrier way, for instance in an experiment:

```python
pt.Experiment(
    [anceretr], 
    dataset.get_topics(), 
    dataset.get_qrels(), 
    eval_metrics=["map"]
)
```

You can also use ANCE as a re-ranker to score text (e.g., as a re-ranker) using `ANCETextScorer`.

```python
ance_text_scorer = pyterrier_ance.ANCETextScorer("/path/to/checkpoint")
# You'll need to use this in a retrieval pipeline that includes the document text, e.g.:
# bm25 >> pt.text.get_text(dataset, 'text') >> ance_text_scorer
```

## Documents longer than Passages

If your documents are longer than passages, you should apply passaging to them before indexing:

```python

# indexing
dataset = pt.get_dataset("irds:vaswani")
import pyterrier_ance
indexer = pt.text.sliding("text", prepend_attr=None) >> pyterrier_ance.ANCEIndexer("/path/to/checkpoint", "/path/to/anceindex")
indexer.index(dataset.get_corpus_iter())

# retrieval 

ance_maxp = pyterrier_ance.ANCERetrieval("/path/to/checkpoint", "/path/to/anceindex") >> pt.text.max_passage()

```

## Examples

Checkout out the notebooks, even on Colab:

 - [Vaswani Corpus](pyterrier_ance_vaswani.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/terrierteam/pyterrier_ance/blob/master/pyterrier_ance_vaswani.ipynb)

The [Terrier data repository](http://data.terrier.org/) contains ANCE indices for several corpora, including Vaswani and MSMARCO Passage v1.

## Implementation Details

We use a [fork-ed copy of ANCE](https://github.com/cmacdonald/ANCE/) that makes it pip installable, and addresses other quibbles.

## References

  - [Xiong20] Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval. Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, Arnold Overwijk. https://arxiv.org/pdf/2007.00808.pdf
  - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

- Craig Macdonald, University of Glasgow
- Nicola Tonellotto, University of Pisa
