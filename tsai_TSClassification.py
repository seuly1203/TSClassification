import pandas as pd
import collections
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from tsai.all import *

train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')

label_counts = collections.Counter(train['leaktype'])
print('Counts by label:', dict(label_counts))
print(f'Naive Accuracy: {100*max(label_counts.values())/sum(label_counts.values()):0.2f}%')

# train 데이터 전처리 -> features(X), target(y)으로 나누기
X = train.drop('leaktype', axis=1)
y = train[['leaktype']]
print(f'X:\n{X}')
print(f'y:\n{y}')

# X normalization
X = np.array(X)
sc = StandardScaler().fit(X.T) # 열방향으로 scaling하기 위해 transpose 사용
X_scaled = sc.transform(X.T).T
print(f'X_scaled:\n{X_scaled}')

# X 3-d array로 만들기
X_scaled = to3d(X_scaled)
print(X_scaled.shape)

# y np.array로 만들기
y = np.array(y['leaktype'])

# splits 생성
splits = get_splits(y, valid_size=0.2, stratify=True, random_state=42, shuffle=True)



# prepare dataloaders
tfms = [None, TSClassification()] # TSClassification == Categorize
batch_tfms = TSStandardize()
dls = get_ts_dls(X_scaled, y, splits=splits, tfms=tfms, batch_tfms=batch_tfms, bs=[64, 128])
print(f'dls.dataset:\n{dls.dataset}')

dls.show_batch(sharey=True) # 데이터 평태 그래프로 보여줌
plt.show()

# build learner
model = build_ts_model(MLSTM_FCN, dls=dls) # MLSTM_FCN자리에 tsai에서 제공하는 모델로 바꿔가며 돌리면 됨
learn = Learner(dls, model, metrics=accuracy)

# learning rate curve
learn.lr_find()

# train
learn = ts_learner(dls, metrics=accuracy, cbs=ShowGraph())
learn.fit_one_cycle(10, lr_max=1e-3)

# 모델 저장
PATH = Path('./models/MLSTM_FCN.pkl')
PATH.parent.mkdir(parents=True, exist_ok=True)
learn.export(PATH)

# visualize data
learn.show_results(sharey=True)
learn.show_probas()

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.most_confused(min_val=3)
interp.print_classification_report()

# create predictions
PATH = Path('./models/MLSTM_FCN.pkl')
learn_gpu = load_learner(PATH, cpu=False)

# gpu, many samples
probas, _, preds = learn_gpu.get_X_preds(X[splits[1]])
print(preds[-10:])


print(accuracy_score(y[splits[1]], preds))