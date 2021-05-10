# Description
**Pfam32.0 classifier.** :dna: :test_tube: :mag:

This is a **DeepChain app** to predict the _protein family id_ out of a given sequence.

For extensive details, check the [deepchain-app-pfam-32.0](https://github.com/theomeb/deepchain-app-pfam-32.0) github repo. :nerd_face:

## Data
- This app has been trained with the _pfam32.0_ dataset available with the [bio-datasets](https://pypi.org/project/bio-datasets) API:
```python
# Load pfam dataset
pfam_dataset = load_dataset("pfam-32.0", force=True)
_, y = pfam_dataset.to_npy_arrays(input_names=["sequence"], target_names=["family_id"])
```

- This dataset contains roughly 1339k protein sequences for which the following features are available:
  - _sequence_ - raw sequence **feature**
  - _sequence_name_ - name of the sequence
  - _split_ - original train/dev/test split
  - _family_id_ - **target**
  - _family_accession_ - associated to _family_id_
  
- There are **17929 unique families**, for which only 13071 are present in all splits.
  
- For the _sequence_ feature, corresponding [ProtBert](https://github.com/agemagician/ProtTrans) (_pooling: mean_) embeddings have been computed. For compute reasons, only the embeddings for the first 200 000 sequences are available. The rest will follow very soon.
- This app used [bio-transformers](https://pypi.org/project/bio-transformers/) to compute these embeddings.

- The original dataset can be found here: [Pfam32.0](ftp://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam32.0/Pfam-A.seed.gz), or on [Kaggle](https://www.kaggle.com/googleai/pfam-seed-random-split).

## Model

![Architecture](https://raw.githubusercontent.com/theomeb/deepchain-app-pfam-32.0/develop/src/architecture.png)

- The classifier takes as input the sequence embeddings (_1024-dim_ vector) and then uses a Dense multi-classification to predict the _protein family id_. The model architecture can be found below:

```python
FamilyMLP(
  (_model): Sequential(
    (0): Linear(in_features=1024, out_features=256, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.1, inplace=False)
    (3): Linear(in_features=256, out_features=256, bias=True)
    (4): ReLU()
    (5): Dropout(p=0.1, inplace=False)
    (6): Linear(in_features=256, out_features=num_classes, bias=True)
  )
)
```

- The model has been trained on the first 200 000 sequences, for which we have embeddings so far, and only on CPU. **A complete model trained on a GPU and the full dataset should be available very soon!** :rocket:

## App structure

- deepchain-app-pfam-32.0
  - src/
    - app.py
    - DESCRIPTION.md
    - tags.json
    - Optionnal : requirements.txt (for extra packages)
  - checkpoint/
    - family_model.pt
    - label_encoder.joblib


This app is mean to be deployed in deepchain.bio and has been implemented thanks to the following libraries:
- The main [deepchain-apps](https://pypi.org/project/deepchain-apps/) package - can be found on pypi.
- The [bio-transformers](https://pypi.org/project/bio-transformers/) package.
- The [bio-datasets](https://pypi.org/project/bio-datasets) package.

## Examples

_compute_scores()_ returns a dictionary for each sequence with the predicted _protein_family_id_.


```python
[
  {
    'protein_family_id': 'PuR_N'
  },
   {
    'protein_family_id':'Rrf2'
  }
]
```

## Author
- Théomé Borck | theome.borck@student.ecp.fr
- _MSc Thesis intern @ [InstaDeep](https://www.instadeep.com/)_
- _Student @ [TUM](https://www.tum.de/en/about-tum/our-university/) & [Centrale Paris](https://www.centralesupelec.fr/)_


## Tags

### libraries
- pytorch==1.5.0

### tasks
- transformers
- supervised
- multi-classification
- protein-family-prediction

### embeddings
- ProtBert (mean)

### datasets
- pfam-32.0 dataset

## License
Apache License Version 2.0
