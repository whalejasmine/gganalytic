# Google Analytics Customer Revenue Prediction

- [Competition Site](https://www.kaggle.com/c/ga-customer-revenue-prediction)


## Requirements
- LightGBM
- XGBoost
- feather-format


## Run Single Model

```bash
cd feature
python feature_extnl.py
python cat2vec_fn.py
cd ../model
python model_lgb_m2.py
```
