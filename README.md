# MSBD6000H Covid19 NER

This repo is for the kaggle inclass project ["ML4NLP-Covid19 NER"](https://www.kaggle.com/c/ml4nlp-covid19ner/overview)

## Dependency
 - pytorch='1.8.0'
 - numpy
 - pandas
 - scikit-learn='0.22.2'

## Model architecture
![alt text](doc/lstm_model.png)  

## To reproduce the results
In src/bilstm.py, set training_mode = False in config class  

```bash
cd src
python bilstm.py
```
The result is stored in 'test_pred.csv'   
