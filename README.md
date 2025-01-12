# IMAGO
IMAGO: An Improved Model Based on Attention Mechanism for Enhanced Protein Function Prediction


## 1. Data preprocessing

We can get the dataset in the paper from the [_CFAGO_](http://bliulab.net/CFAGO/static/dataset/Dataset.rar) website and save them in the `data` folder. 

```

python annotation_preprocess.py 
python network_data_preprocess.py
python attribute_data_preprocess.py

```


## 2. Self supervised leaning


```

python self_supervised_leaning.py

```


## 3. Fine-Tuning


```

python IMAGO.py

```




