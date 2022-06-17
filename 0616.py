import pandas as pd
import collections
import numpy as np
from tsai.all import *
from collections import Counter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# PREPARE DATA
train = pd.read_csv('./data/train.csv')
#test = pd.read_csv('./leaktype_test.csv')
X = np.array(train.drop('leaktype', axis=1))
y = np.array(train.leaktype) # y완
print(y[:10])
print(X.shape)
X_1d = X.reshape(1, -1) # normalization 위해 1-d 배열로 변경
print(X_1d.shape)

# X_1d

# train 데이터 label 별 갯수 파악
label_counts = collections.Counter(train['leaktype'])
print('Counts by label:', dict(label_counts))
print(f'Naive Accuracy: {100*max(label_counts.values())/sum(label_counts.values()):0.2f}%')


# NORMALIZATION
scaler = MinMaxScaler()
scaler = scaler.fit(X_1d.T)
normalized_X = scaler.transform(X_1d.T).T
norm_X = normalized_X.reshape((33600, 513)) # 데이터 별로 변경 필요
norm_X = to3d(norm_X) # X완

# TRAIN

# splits 생성
X, y = norm_X, y

model_name = 'InceptionTimePlus'
data_type = 'before-aug'

splits = get_splits(y, valid_size=0.2, stratify=True, random_state=42, shuffle=True) ################


# prepare dataloaders
tfms = [None, TSClassification()] # TSClassification == Categorize
batch_tfms = TSStandardize()
dls = get_ts_dls(X, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
print(f'dls.dataset:\n{dls.dataset}')
dls.show_batch(sharey=True) # 데이터 그래프로 보여줌
plt.show()

# build learner
model = build_ts_model(InceptionTimePlus, dls=dls) # model
learn = Learner(dls, model, metrics=accuracy)

# learning rate curve
learn.lr_find()

# train
learn = ts_learner(dls, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(10, lr_max=1e-3)

# 모델 저장
PATH = Path(f'./models/{model_name}_{data_type}.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)

# visualize data
learn.show_results(sharey=True)
learn.show_probas()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.most_confused(min_val=3)
interp.print_classification_report()
plt.show()

# create predictions
PATH = Path(f'./models/{model_name}_{data_type}.pkl')
learn_gpu = load_learner(PATH, cpu=False)
probas, _, preds = learn_gpu.get_X_preds(X[splits[0]])
print(preds[-10:])