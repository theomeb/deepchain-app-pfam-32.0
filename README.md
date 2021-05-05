# Pfam32.0 classifier

## Description
This is a **DeepChain app** to predict the _protein family id_ out of a given sequence. 

- This app has been trained with the Pfam32.0 dataset which you can pull 
with the [bio-datasets](https://pypi.org/project/bio-datasets):
```python
# Load pfam dataset
pfam_dataset = load_dataset("pfam-32.0", force=True)
_, y = pfam_dataset.to_npy_arrays(input_names=["sequence"], target_names=["family_id"])
```

- This app uses [ProtBert](https://github.com/agemagician/ProtTrans) to compute embeddings with the `mean` pooling.
This is done thanks to [bio-transformers](https://pypi.org/project/bio-transformers/)
  
- The classifier then computes the protein family id thanks to a Dense classification on top of the sequences' embeddings. 


This app is mean to be deployed in deepchain.bio and has been implemented thanks to the following libraries:
- The main [deepchain-apps](https://pypi.org/project/deepchain-apps/) package - can be found on pypi.
- The [bio-transformers](https://pypi.org/project/bio-transformers/) package.
- The [bio-datasets](https://pypi.org/project/bio-datasets) package.

### App structure

- my_application
  - src/
    - app.py
    - DESCRIPTION.md
    - tags.json
    - Optionnal : requirements.txt (for extra packages)
  - checkpoint/
    - Optionnal : family_model.pt
    - Optionnal : label_encoder.joblib
  

## Examples

`compute_scores()` returns a dictionary for each sequence with the predicted. `"protein_family_id"` 


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

## Templates

Further information on DeepChain App templates can be found [here](./README_deepchainapps.md).

## License
Apache License Version 2.0
