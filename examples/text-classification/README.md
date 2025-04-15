## Natural Language Inference (NLI)
Download the AmericasNLI test data using the below comand:

```
cd data
git clone https://github.com/nala-cub/AmericasNLI.git
```
Use train_nli.sh script to train DeFT-X task vector for NLI and eval_nli.sh to evaluate on the AmericasNLI test data.

## Sentiment Analysis (SA)
To create a version of the SMSA dataset with examples from the test set of NusaX-senti removed, run

```
# first run "pip install editdistance" and "pip install nltk" if you don't already have it installed
python map_nusa.py
```

Use train_sa.sh script to train DeFT-X task vector for SA and eval_sa.sh to evaluate on the NusaX test data.