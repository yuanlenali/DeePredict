# DeePredict

DeePredict implements state-of-art Deep Neural Network based algorithm for CTR Prediction. 

## Dataset
- The algorithms use Criteo (Kaggle) dataset, which contains about 45 million records. There are 13 numerical (dense) features and 26 categorical (sparse) features.
- dataset link: http://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/

## Models
 - ##### DeepFM: Factorization-Machine based Neural Network
    - DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    - paper link: https://www.ijcai.org/proceedings/2017/0239.pdf
    - How To Run: Run data/deepFM_dataPreprocess.py to pre-process the data, then run model/deepFM.py.

 - ##### DCN: Deep and Cross Neural Network
    - Deep & Cross Network for Ad Click Predictions
    - paper link: https://arxiv.org/abs/1708.05123
    - How To Run: Run data/deepCrossNet_dataPreprocess.py to pre-process the data, then run model/deepCrossNet.py.

- ##### xDeepFM: to do