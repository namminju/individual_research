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

## 2) GridSearchCV에서 사용할 모델별 파라미터 후보
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



## 3) 데이터 읽어오기
file_path = 'C:/Users/user/Desktop/개별연구/개별연구_2022/PCME/PCME/cancer.csv'

if os.path.isfile(file_path):
    data = pd.read_csv(file_path, na_values="#DIV/0!", encoding='cp949')
    print("파일을 읽어옵니다.")
else:
    print("cancer.csv 파일을 입력해주세요.")
    sys.exit()



##null 제거

df_checknull = pd.DataFrame(data.isnull().sum())#각 열의 null 개수 데이터프레임
df_havenull = df_checknull[df_checknull[0]!=0]# null 값이 있는 열 데이터프레임
havenull_columns = list(df_havenull.index)# null 값이 있는 열의 이름 리스트

for col in havenull_columns:#null 값이 있는 열들에 대해서
    data.loc[(data[col].isnull()) & (data['label']==1),col] = data[data['label']==1][col].mean()
    #label이 1이면 label이 1인 행들 중 해당 열의 평균값으로 대체
    data.loc[(data[col].isnull()) & (data['label']==0),col] = data[data['label']==0][col].mean()
    #label이 0이면 label이 0인 행들 중 해당 열의 평균값으로 대체
print("null존재 여부:",data.isnull().values.any())#null이 모두 제거되었는 지 확인

## +)정규화
scaler = MinMaxScaler()
scaler.fit(data.iloc[:, :-1])
data_scaled = scaler.transform(data.iloc[:, :-1])
data_scaled = pd.DataFrame(data=data_scaled, columns=data.iloc[:, :-1].columns)

new_df = pd.concat([data.loc[:, 'label'], data_scaled], axis=1)

print("정규화 완료")

## +burutapy
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
boruta = BorutaPy(rf, n_estimators=200, verbose=2, random_state=42,)

stats=data.iloc[:, :].describe()

X = new_df.iloc[:, :-1].values
y = new_df.loc[:, 'label']
print("data set 설정 완료")
print(X.shape)
# PCA 수행
pca = PCA()
X_pca = pca.fit_transform(X)
print("pca 완료")

# PCA 변환된 데이터를 DataFrame으로 변환
selected_df = pd.DataFrame(data=X_pca, columns=[f"PC_{i+1}" for i in range(X_pca.shape[1])])

# Reset index of selected_df
selected_df.reset_index(drop=True, inplace=True)
print("borutapy 시작")
## +burutapy
boruta.fit(selected_df.values, y)
print("borutapy 완료")

# 결과 출력

##selected_features_name = selected_df.columns[boruta.support_]
rankings = list(zip(selected_df.columns[:-1], boruta.ranking_))
selected_features = selected_df[boruta.support_]
selected_features_name = selected_df.columns[boruta.support_].values.tolist()
num_selected_features = len(selected_features_name)
print("결과: ", selected_features_name)

##몇 개의 feature를 이용할지 입력받기
print(num_selected_features,"개의 요소 중")
number = int(input("몇 개의 feature를 이용할까요: "))
# 중요도에 따라 상위 number개의 특징을 선택합니다.
top_n_features = [rank[0] for rank in sorted(rankings, key=lambda x: x[1])][:number]

selected_df = selected_df[top_n_features]
selected_df = pd.concat([selected_df,y], axis=1)#정규화한 feature와 label 합치기
markers_1 =[top_n_features]
print(markers_1 )
X_train, X_test, y_train, y_test = train_test_split(selected_df.iloc[:, :-1], selected_df.loc[:, 'label'], test_size=0.2, random_state=42)


#sys.exit()
#####여기부터 수정

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

'''
dfs : 환자 샘플 데이터셋을 getPC_df_othdisease의 옵션에 따라 분류한 데이터셋
csv_names : getPC_df_othdisease의 RPC/low_ca19_9/normal/pds/oc 옵션에 따라 결과값을 저장할 csv 파일명
markse_sets : dfs의 각 샘플 데이터셋의 train에 쓰일 마커셋
models/model_params : dfs와 marker_sets의 조합을 훈련시킬 학습모델로, train_evaluate의 GridSearchCV=True일 경우 model_params의
                      파라미터 후보를 gridsearch하여 최적의 파라미터를 찾고, False일 경우 설정 파라미터로 훈련시킨다.
'''
## Final) main 코드
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
    '''
    컬럼명의 의미는 각각
    마커, 모델, 훈련/테스트 데이터여부와 train_evaluate의 results_list의 sensitivity, speicficity, auc, youden, cutoff기준,
    현재 적용모델의 파라미터 순이다.
    '''
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
            y_pred = model.predict(X_test)  # 테스트 데이터에 대한 예측
            TP = sum((y_test == 1) & (y_pred == 1))  # True Positive 계산
            FN = sum((y_test == 1) & (y_pred == 0))  # False Negative 계산
            TN = sum((y_test == 0) & (y_pred == 0))  # True Negative 계산
            FP = sum((y_test == 0) & (y_pred == 1))  # False Positive 계산
            sensitivity = calculate_sensitivity(TP, FN)  # 민감도 계산
            specificity = calculate_specificity(TN, FP)  # 특이도 계산
            youden_index = calculate_youden_index(sensitivity, specificity)  # Youden 지수 계산
            if youden_index>best_youden:
                best_youden=youden_index
            print("Youden Index of ",model," :", youden_index)
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
                    
        '''
        for loop를 통해 샘플 데이터셋 / 마커셋 / 학습모델 별 결과를 도출한다.
        enumerate(markerset) : i, marker에 각각 인덱스와 markerset의 marker가 차례대로 저장된다.
        results_list : getResults_changing_cutoff의 return인 results(sensitivity, speicficity, auc, youden, cutoff기준)
        best_param : GridSearchCV=True인 경우 grid_model.best_params_가 best_param이 되고, False일 경우
                     model_params의 default parameter이다.

        df_perf의 'Markers', '# of Markers', 'Model', 'Train/Test', 'Sensitivity', 'Specificity', 'AUC', 'Youden',
        'Cutoff Point', 'model_param'이 각각 result[0] ~ result[9]에 대응된다.

        for loop에서 비교하는 행의 model_param이 best_param일 경우
        초기에 0으로 설정된 max_auc, max_youden을 각각 result[6],[7]에서 한 행씩 비교하여
        auc, youden이 각각 가장 높았던 샘플군 별 학습모델/cutoff를 max_auc, max_youden에 갱신하고 그 때의 학습모델을
        max_auc_model과 max_youden_model에 저장한다.
        '''

    df_perf.to_excel(csv_name + '.xlsx', index=False, header=False)  ## 샘플군마다 csv 이름 바꾸기

    
sys.exit()
    



'''
    csv_name(샘플 df명) 파일명으로, 즉 getPC_df_othdisease으로 분류된 샘플군 별로 df_perf
    ('Markers', '# of Markers', 'Model', 'Train/Test', 'Sensitivity', 'Specificity', 'AUC', 'Youden', 'Cutoff Point')
    행들을 xlsx형식으로 저장한다.
    '''

#     X_train, X_test, y_train, y_test  = train_test_split(
#         X_CA199, y_CA199, test_size=0.2, stratify=y_CA199, random_state=42
#     )
#     fpr_train, tpr_train, _ = roc_curve(y_train, X_train)
#     fpr_test, tpr_test, _ = roc_curve(y_test, X_test)
#     auc_train = auc(fpr_train, tpr_train)
#     auc_test = auc(fpr_test, tpr_test)

#     ax_auc_train.plot(fpr_train, tpr_train, 'r-', label='CA19-9 Only (AUC = %0.2f)' % auc_train)
#     ax_auc_test.plot(fpr_test, tpr_test, 'r-', label='CA19-9 Only (AUC = %0.2f)' % auc_test)
#     for best_model in best_auc_model:
#         train_roc, test_roc = main(best_model[0], best_model[1], best_model[2],
#                                    save_plot_as=best_model[3], ax_train=ax_auc_train, ax_test=ax_auc_test)

#     ax_you_train.plot(fpr_train, tpr_train, 'r-', label='CA19-9 Only (AUC = %0.2f)' % auc_train)
#     ax_you_test.plot(fpr_test, tpr_test, 'r-', label='CA19-9 Only (AUC = %0.2f)' % auc_test)
#     for best_model in best_youden_model:
#         train_roc, test_roc = main(best_model[0], best_model[1], best_model[2],
#                                    save_plot_as=best_model[3], ax_train=ax_you_train, ax_test=ax_you_test)