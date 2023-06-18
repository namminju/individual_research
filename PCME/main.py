## main
from util import getResults_changing_cutoff, train_evaluate,calculate_sensitivity,calculate_specificity,calculate_youden_index
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import os, sys
import pandas as pd
import matplotlib.pyplot as plt

from warnings import simplefilter, filterwarnings
from sklearn.exceptions import ConvergenceWarning

filterwarnings("ignore", category=ConvergenceWarning)
filterwarnings("ignore")
from tqdm import tqdm
from boruta import BorutaPy
import numpy as np
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler

print("프로그램 실행 시작")

## GridSearchCV에서 사용할 모델별 파라미터 후보
param_lr = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
            "penalty" : ["l2"],
            'class_weight':[{0:0.1, 1:0.9}],
            'random_state':[42, 99]}

param_rf = {'n_estimators': [100, 500],
            'max_depth': [None, 3, 5, 7, 9, 11],
#             'min_samples_split' : [2, 5],
            'class_weight':[{0:0.1, 1:0.9}],
            'random_state':[42, 99]}

param_lgbm = {'n_estimators': [50, 100, 500],
              'max_depth':[-1, 5, 7, 9, 11],
#               'num_leaves':[40, 60, 80],
#               'min_child_samples':[5,10,15],
              'class_weight':[{0:0.1, 1:0.9}],
              'reg_alpha':[0, 0.01],
              'random_state':[42, 99]}

param_xgb = {'n_estimators': [50, 100, 500],
             'max_depth':[-1, 3, 5, 7, 9, 11],
#              'num_leaves':[30, 60],
#              'gamma': [0.5, 1, 2],
             'class_weight':[{0:0.1, 1:0.9}],
             'random_state':[42, 99]}



##  데이터 읽어오기
file_path = './cancer.csv'

if os.path.isfile(file_path):
    data = pd.read_csv(file_path, na_values="#DIV/0!", encoding='cp949')
    print("파일을 읽어옵니다.")
else:
    print("cancer.csv 파일을 입력해주세요.")
    sys.exit()



# Null 값을 해당 컬럼의 평균 값으로 대체하여 제거

df_checknull = pd.DataFrame(data.isnull().sum())#각 열의 null 개수 데이터프레임
df_havenull = df_checknull[df_checknull[0]!=0]# null 값이 있는 열 데이터프레임
havenull_columns = list(df_havenull.index)# null 값이 있는 열의 이름 리스트

for col in havenull_columns:#null 값이 있는 열들에 대해서
    data.loc[(data[col].isnull()) & (data['label']==1),col] = data[data['label']==1][col].mean()
    #label이 1이면 label이 1인 행들 중 해당 열의 평균값으로 대체
    data.loc[(data[col].isnull()) & (data['label']==0),col] = data[data['label']==0][col].mean()
    #label이 0이면 label이 0인 행들 중 해당 열의 평균값으로 대체
print("null존재 여부:",data.isnull().values.any())
# null이 모두 제거되었는 지 확인

## 데이터 정규화
scaler = MinMaxScaler()
scaler.fit(data.iloc[:, :-1])
data_scaled = scaler.transform(data.iloc[:, :-1])
data_scaled = pd.DataFrame(data=data_scaled, columns=data.iloc[:, :-1].columns)

new_df = pd.concat([data.loc[:, 'label'], data_scaled], axis=1)

print("정규화 완료")

stats=data.iloc[:, :].describe()

X = new_df.iloc[:, :-1].values
y = new_df.loc[:, 'label']
print("data set 설정 완료")
print(X.shape)

pca_run = input("PCA를 진행할까요?(y/n): ")
if pca_run == 'y':
    # PCA 수행
    pca = PCA()
    X_pca = pca.fit_transform(X)
    print("pca 완료")
    # PCA 변환된 데이터를 DataFrame으로 변환
    selected_df = pd.DataFrame(data=X_pca, columns=[f"PC_{i+1}" for i in range(X_pca.shape[1])])
    # Reset index of selected_df
    selected_df.reset_index(drop=True, inplace=True)
else:
    print("pca를 진행하지 않습니다.")
    selected_df = new_df.iloc[:, :-1]

boruta_num = int(input("borutapy 진행 횟수를 입력해주세요: "))
## +burutapy
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta = BorutaPy(rf, n_estimators='auto', max_iter=boruta_num, verbose=2, random_state=42)
print("borutapy 시작")
## +burutapy
boruta.fit(selected_df.values, y)
print("borutapy 완료")

# 결과 출력

##selected_features_name = selected_df.columns[boruta.support_]
rankings = list(zip(selected_df.columns[:-1], boruta.ranking_))
selected_features = selected_df.iloc[:, boruta.support_]
selected_features_name = selected_df.columns[boruta.support_].values.tolist()
num_selected_features = len(selected_features_name)
print("결과: ", selected_features_name)

##몇 개의 feature를 이용할지 입력받기
print(num_selected_features, "개의 요소 중")
number = int(input("몇 개의 feature를 이용할까요: "))
# 중요도에 따라 상위 number개의 특징을 선택합니다.
top_n_features = [rank[0] for rank in sorted(rankings, key=lambda x: x[1])][:number]

selected_df = selected_df[top_n_features]
selected_df = pd.concat([selected_df, y], axis=1)  # 정규화한 feature와 label 합치기
markers_1 = [top_n_features]
print(markers_1)

X_train, X_test, y_train, y_test = train_test_split(selected_df.iloc[:, :-1], selected_df.loc[:, 'label'], test_size=0.2, random_state=42)


#sys.exit()

## 4) train-evaluate 하고자 하는 샘플군, 샘플군 이름, 마커셋, 모델 -> list
dfs = [
       (selected_df)
      ]

csv_names = [
             'normal vs. Cancer'
            ]

markers_sets = [
                markers_1
               ]

models = [
          LogisticRegression(class_weight={0:0.1, 1:0.9}, random_state=42, n_jobs=-1),
          RandomForestClassifier(n_estimators=500, max_depth=5, class_weight={0:0.1, 1:0.9}, random_state=99),
          LGBMClassifier(n_estimators=500, max_depth=5, class_weight={0:0.1, 1:0.9}, random_state=42),
          XGBClassifier(n_estimators=500, max_depth=5, class_weight={0:0.1, 1:0.9}, random_state=42)
         ]

model_params = [
                param_lr,
                param_rf,
                param_lgbm,
                param_xgb
               ]



# for loop -> 각 샘플군에서 여러개 마커셋을 여러개 모델로 train-evaluate
best_auc_models, best_youden_models = [], []

for dataset, markerset, csv_name in zip(dfs, markers_sets, csv_names):
    if not os.path.isdir(csv_name):
        os.mkdir(csv_name)
    print('Dataset :', csv_name)
    print('Dataset shape :', dataset.shape)

    df_perf = pd.DataFrame(
        columns=['Markers', '# of Markers', 'Model', 'Train/Test', 'Sensitivity', 'Specificity', 'AUC', 'Youden',
                 'Cutoff Point', 'model_param'])

    best_youden=0
    ## 샘플군마다 여러개 마커셋 실험
    for i, marker in enumerate(markerset):
        print(*marker)
        max_auc, max_youden = 0, 0
        # 각 마커셋마다 여러개 모델 실험
        for model, model_param in zip(models, model_params):
            print(model)
            results_list, best_param = train_evaluate(marker, model, dataset, model_param=model_param,
                                              gridsearchcv=False, save_plot_as=csv_name)
            model.fit(X_train, y_train)  # 모델 학습

            ## result processing
            for result in results_list:
                result[0] = ' '.join(marker)
                result[1] = len(marker)
                result[2] = str(model).split('(')[0]
                try:
                    result[9] = best_param
                except:
                    pass

                df_perf = df_perf.append(pd.Series(result), ignore_index=True)

                if result[6] > max_auc and result[3] == 'test':
                    max_auc_model = model
                    max_auc = result[6]
                if result[7] > max_youden and result[3] == 'test':
                    max_youden_model = model
                    max_youden = result[7]
                    

    df_perf.to_excel(csv_name + '.xlsx', index=False, header=False)  ## 샘플군마다 csv 이름 바꾸기

 