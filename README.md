## Source codes for ``Efficient Classification of Long Documents Using Transformers''

Please refer to our paper for more details and cite our paper if you find this repo useful:

```
@inproceedings{park-etal-2022-efficient,
    title = "Efficient Classification of Long Documents Using Transformers",
    author = "Park, Hyunji  and
      Vyas, Yogarshi  and
      Shah, Kashif",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-short.79",
    doi = "10.18653/v1/2022.acl-short.79",
    pages = "702--709",
}
```

## Instructions

### 1. Install required libraries

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Prepare the datasets

#### Hyperpartisan News Detection 

* Available at <https://zenodo.org/record/1489920#.YLferh1Olc8>
* Download the datasets

```
mkdir data/hyperpartisan
wget -P data/hyperpartisan/ https://zenodo.org/record/1489920/files/articles-training-byarticle-20181122.zip
wget -P data/hyperpartisan/ https://zenodo.org/record/1489920/files/ground-truth-training-byarticle-20181122.zip
unzip data/hyperpartisan/articles-training-byarticle-20181122.zip -d data/hyperpartisan
unzip data/hyperpartisan/ground-truth-training-byarticle-20181122.zip -d data/hyperpartisan
rm data/hyperpartisan/*zip
```
  
*  Prepare the datasets with the resulting xml files and this preprocessing script (following [Longformer](https://arxiv.org/abs/2004.05150)): <https://github.com/allenai/longformer/blob/master/scripts/hp_preprocess.py>

#### 20NewsGroups

* Originally available at <http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz>
* Running `train.py` with the `--data 20news` flag will download and prepare the data available via `sklearn.datasets` (following [CogLTX](https://proceedings.neurips.cc/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf)).
We adopt the train/dev/test split from [this ToBERT paper](https://ieeexplore.ieee.org/document/9003958).
  
#### EURLEX-57K

* Available at <https://github.com/iliaschalkidis/lmtc-emnlp2020>
* Download the datasets

```
mkdir data/EURLEX57K
wget -O data/EURLEX57K/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/EURLEX57K/datasets.zip -d data/EURLEX57K
rm data/EURLEX57K/datasets.zip
rm -rf data/EURLEX57K/__MACOSX
mv data/EURLEX57K/dataset/* data/EURLEX57K
rm -rf data/EURLEX57K/dataset
wget -O data/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
```

* Running `train.py` with the `--data eurlex` flag reads and prepares the data from `data/EURLEX57K/{train, dev, test}/*.json` files
* Running `train.py` with the `--data eurlex --inverted` flag creates Inverted EURLEX data by inverting the order of the sections
* `data/EURLEX57K/EURLEX57K.json` contains label information.

#### CMU Book Summary Dataset

* Available at <http://www.cs.cmu.edu/~dbamman/booksummaries.html>

```
wget -P data/ http://www.cs.cmu.edu/~dbamman/data/booksummaries.tar.gz
tar -xf data/booksummaries.tar.gz -C data
```

* Running `train.py` with the `--data books` flag reads and prepares the data from `data/booksummaries/booksummaries.txt`
* Running `train.py` with the `--data books --pairs` flag creates Paired Book Summary by combining pairs of summaries and their labels


### 3. Run the models

```
e.g. python train.py --model_name bertplusrandom --data books --pairs --batch_size 8 --epochs 20 --lr 3e-05
```

cf. Note that we use the source code for the CogLTX model: <https://github.com/Sleepychord/CogLTX>

### Hyperparameters used

#### Hyperpartisan

| Parameter  | BERT  | BERT+TextRank | BERT+Random | Longformer                                        | ToBERT |
|------------|-------|---------------|-------------|---------------------------------------------------|--------|
| Batch size | 8     | 8             | 8           | 16                                                | 8      |
| Epochs     | 20    | 20            | 20          | 20                                                | 20     |
| LR         | 3e-05 | 3e-05         | 5e-05       | 5e-05                                             | 5e-05  |
| Scheduler  | NA    | NA            | NA          | [warmup](https://arxiv.org/abs/2004.05150)  | NA     |

#### 20NewsGroups, Book Summary, Paired Book Summary

| Parameter  | BERT  | BERT+TextRank | BERT+Random | Longformer                                        | ToBERT |
|------------|-------|---------------|-------------|---------------------------------------------------|--------|
| Batch size | 8     | 8             | 8           | 16                                                | 8      |
| Epochs     | 20    | 20            | 20          | 20                                                | 20     |
| LR         | 3e-05 | 3e-05         | 3e-05       | 0.005                                             | 3e-05  |
| Scheduler  | NA    | NA            | NA          | [warmup](https://arxiv.org/abs/2004.05150)  | NA     |

#### EURLEX, Inverted EURLEX

| Parameter  | BERT  | BERT+TextRank | BERT+Random | Longformer                                        | ToBERT |
|------------|-------|---------------|-------------|---------------------------------------------------|--------|
| Batch size | 8     | 8             | 8           | 16                                                | 8      |
| Epochs     | 20    | 20            | 20          | 20                                                | 20     |
| LR         | 5e-05 | 5e-05         | 5e-05       | 0.005                                             | 5e-05  |
| Scheduler  | NA    | NA            | NA          | [warmup](https://arxiv.org/abs/2004.05150)        | NA     |



