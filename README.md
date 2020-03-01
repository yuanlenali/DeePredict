# DeePredict

DeePredict contains the implementation of bunch of DeepNN based algorithm for CTR Prediction. 

## Dataset
The algorithms use Criteo dataset, which contains about 45 million records. There are 13 numerical(dense) features and 26 categorical(sparse) features.
dataset link: http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/

## Models
 - ##### DeepFM: Factorization-Machine based Neural Network
    - paper link: https://www.ijcai.org/proceedings/2017/0239.pdf
    - How To Run:
    Run data/deepFM_dataPreprocess.py to pre-process the data, then run model/deepFM.py.