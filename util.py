import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split, cross_val_score)
from sklearn.metrics import (
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    make_scorer,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc
)
from scikitplot.metrics import plot_roc_curve
import os, sys
import pandas as pd
import matplotlib.pyplot as plt
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warnings import simplefilter, filterwarnings
from sklearn.exceptions import ConvergenceWarning
from scipy.integrate import simps
filterwarnings("ignore", category=ConvergenceWarning)
filterwarnings("ignore")
from tqdm import tqdm
import datetime as dt

def calculate_sensitivity(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)

def calculate_specificity(true_negative, false_positive):
    return true_negative / (true_negative + false_positive)

def calculate_youden_index(sensitivity, specificity):
    return sensitivity + specificity - 1



def train_evaluate(marker, model, dataset, gridsearchcv=True, model_param=None, save_plot_as=None, kfoldcv=False,
                   ax_train=None, ax_test=None):
    df = dataset

    X = df.loc[:, marker]
    y = df.loc[:, 'label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_without = X_train
    X_test_without = X_test

    '''
    df : getPC_df_othdisease에서 리턴된 샘플군
    X : 샘플군의 훈련에 쓰일 마커셋
    y : getPC_df_othdisease에서 분류된 질병 클래스
    train_test_split을 통해 train/test 데이터를 8:2 로 나눈다.
    stratify=y를 통해 train/test 데이터의 클래스값이 일정 비율로 나오게한다.
    '''
    sen_score = make_scorer(recall_score, pos_label=1)
    spe_score = make_scorer(recall_score, pos_label=0)

    ## Grid Search CV
    if gridsearchcv == True:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
        grid_model = GridSearchCV(estimator=model,
                                  param_grid=model_param,
                                  cv=cv,
                                  scoring='roc_auc',
                                  n_jobs=-1,
                                  verbose=2)
        grid_model.fit(X_train, y_train)
        clf = grid_model
        print(clf)
        clf.fit(X_train, y_train)
        '''
        main에서 train_evaluate의 gridsearchcv값을 True(Default)로 하였을 때
        StratifiedKFold를 통해 main에서 지정한 학습모델의 파라미터 후보들을 교차검증하여 최적의 후보를 도출한다.
        StratifiedKFold는 n_splits값만큼 나눠진 데이터셋의 클래스값을 일정 비율로 나눈다.
        설정된 grid_model과 train 데이터를 통해 fit으로 훈련시킨다.
        estimator : 학습모델
        param_grid : 파라미터후보군
        scoring='roc_auc' : 파라미터의 성능척도를 ROC곡선의 AUC값으로 한다.
        n_jobs : 연산에 쓰일 코어의 수
        verbose : 로그 출력 여부 설정(2 : 함축적 정보)
        '''

        model_param = grid_model.best_params_
        # performance of each parameter set
        cv_result = pd.DataFrame(grid_model.cv_results_)
        cv_result.to_csv(str(dt.datetime.now().strftime('%m-%d %H-%M-%S ')) + str(model).split('(')[0] + '.csv')
        # save best parameter for each model as txt file
        with open("BEST_PARAMS.txt", "a") as f:
            f.write(str(dt.datetime.now()) + '\n')
            f.write(str(X.columns) + '\n')
            f.write(str(model).split('(')[0] + '\n')
            f.write(str(grid_model.best_params_) + '\n')
            f.write('{:.4f}'.format(grid_model.best_score_) + '\n')
            f.write('=' * 10 + '\n')
            '''
            grid_model.best_params_ : main의 학습모델의 파라미터 후보 중 auc 값이 가장 높은 파라미터 조합을 model_param에 저장한다.

            cv_result는 파라미터 조합 별 grid search의 결과를 담은 grid_model.cv_results_를 DataFrame형으로 변환한 것으로 to_csv를 통해
            '시간'_'적용모델'.csv의 형태로 저장한다.
            BEST_PARAMS.txt엔 date/markerset/model/best parameter set/score(auc) 값을 시간순으로 저장한다.
            '''
    else:
        clf = model
        clf.fit(X_train_without, y_train)
    '''
    GridSearchCV = False일 경우 파라미터 후보를 이용하지 않고
    main의 models에서 설정한 파라미터로 X_train/y_train을 바로 학습시킨다. 
    '''

    ## K-fold Cross-Validation : fold 별 ROC curve
    ## 미완성 코드(잘 안 쓰는 코드 - 무시하기)
    
    if kfoldcv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fig, ax = plt.subplots()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
    
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model.fit(X[train_idx], y[train_idx])
            y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y[test_idx], y_pred_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
    
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        ax.plot(mean_fpr, mean_tpr, color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2, alpha=0.8
                )
        ax.legend(loc="lower right")
        plt.show()
        
        # CV의 평균 성능
        spe_score = make_scorer(recall_score, pos_label=0)
        sen_score = make_scorer(recall_score, pos_label=1)
        cv_result = cross_validate(model, X, y, cv=5,
                               scoring={'sen': sen_score, 'spe': spe_score, 'auc': 'roc_auc'})

        sen_list = cv_result['test_sen']
        spe_list = cv_result['test_spe']
        auc_cv = mean_auc
        youden_list = sen_list + spe_list - 1
        youden_cv = np.mean(youden_list)
        spe_cv = np.mean(spe_list)
        sen_cv = np.mean(sen_list)



    ## Final Model Fitting
    # cv==False(Default)
    #else : 
    y_pred_proba_test = clf.predict_proba(X_test_without)[:, 1]
    y_pred_proba_train = clf.predict_proba(X_train_without)[:, 1]
    '''
    clf : X_train/X_train_without와 y_train으로 fit된 grid_model
    y_pred_proba_test/y_pred_proba_train은 predict_proba로 도출된 clf 모델의 클래스 예측 확률의 집합으로
    getResults_changing_cutoff에서 roc_curve에 대한 정보로 사용된다.
    '''

    if ax_train != None and ax_test != None:
        legend = save_plot_as + '\n' + str(model).split('(')[0]  # best_auc_model 등 그릴때 원래 이 코드
        #         legend = save_plot_as
        train_roc = plot_roc(y_train, y_pred_proba_train, legend=legend, ax=ax_train)
        test_roc = plot_roc(y_test, y_pred_proba_test, legend=legend, ax=ax_test)

        return train_roc, test_roc

    ## Results of different Cut-off Points
    # return list of performances at each cut-off points
    else:
        results_list = getResults_changing_cutoff(y_train, y_pred_proba_train,
                                                  y_test, y_pred_proba_test,
                                                  min_senup=[90, 95], min_speup=[90, 95])

        # save plot
        if save_plot_as != None:
            title = ' '.join(marker) + '\n' + str(model).split('(')[0]
            if len(X.columns) > 7:
                title = ' '.join(marker[:7]) + '\n' + ' '.join(marker[7:]) + '\n' + str(model).split('(')[0]

            fig, ax = plt.subplots()

            train_roc = plot_roc(y_train, y_pred_proba_train, legend='train', ax=ax)
            test_roc = plot_roc(y_test, y_pred_proba_test, legend='test', ax=ax)
            plt.title(title)
            plt.savefig(save_plot_as + '/' + ' '.join(marker) + '_' + str(model).split('(')[0] + '.png')
        '''
        getResults_changing_cutoff의 min_senup, min_speup의 최소값 후보, max_youden(default=True)옵션을 통해
        optimal threshold로부터 도출된 train_perf, test_perf의 수치들이 results에 append된 행들이 results_list에 저장된다.

        plot_roc함수를 통해 train 모델과 test 모델의 roc curve 그래프를 표시한다.
        title : marker의 개수가 8개 이상일 경우 줄바꿈을 하고 적용된 모델명을 표시한다.
        legend : plot_roc함수의 RocCurveDisplay.from_predictions의 옵션으로 auc값이 함께 표시된다.
        savefig : save_plot_as(csv_name와 같은 이름의 경로)에 'markerset'_'모델명'.png의 형태로 그래프가 저장된다.
        '''
        return results_list, model_param  # model_param : GridSearchCV result


    '''
    RPC
    False(Default) : PDAC에 해당하는 모든 값을 PC클래스로 이용한다.
    True : PDAC중 RPC만을 이용한다.
    name_pc : 출력할 PC클래스명
    
    normal
    True(Default) : label이 Normal인 샘플만을 normal 클래스로 이용한다.
    False :  

    pds
    'normal' : Pancreatic Disease(pd)를 normal 클래스로 분류
    'pc' : pd를 PC클래스로 분류
    
    oc
    'normal' : Other Cancer를 normal 클래스로 분류
    'pc' : oc를 PC클래스로 분류
    '''


    '''
    앞서 선택된 옵션에 따라 분류된 결과를 통해
    normal로 분류된 label을 가지는 행을 0, PC로 분류된 행을 1로 클래스를 설정한다.
    
    preprocess는 샘플 데이터셋을 normal, PC 순으로 정렬되도록
    pd.DataFrame으로 분리 후 concat으로 묶는다.

    get_loc을 통해 preprocess 데이터셋의 위치를 loc로 반환하여 iloc의 입력값으로 사용한다.
    np.r_을 통해 label, CA19_9 칼럼과 IFNA1~PTGES(마커) 칼럼들의 행들을 모아
    processed에 저장한다.
    '''

    '''
    low_ca19_9 = True일 경우
    drop을 통해 float형으로 변환된 37.0 이상의 CA19_9값을 가지는 행을 삭제하고,
    reset_index을 통해 새로운 인덱스를 배치한다.
    '''



    #     processed['label'] = processed['label'].astype('int64')
    '''
    전처리가 완료된 processed의 데이터의 수를 출력하기 위해 앞서 설정한 클래스(0, 1)에
    따라 normal, PC로 다시 분류하여 len를 통해 출력한다.
    PC의 경우 RPC 옵션에서 설정된 name_PC의 이름을 출력한다.
    
    최종적으로 float형으로 형변환하여 return한다.
    '''




def getResults_changing_cutoff(y_train, y_pred_proba_train, y_test, y_pred_proba_test,
                               min_senup=[], min_speup=[], min_senspe=[]):
    results = []

    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_pred_proba_test)

    sensitivity_train, specificity_train, youden_train = tpr_train, 1 - fpr_train, tpr_train - fpr_train
    sensitivity_test, specificity_test, youden_test = tpr_test, 1 - fpr_test, tpr_test - fpr_test
    auc_train = roc_auc_score(y_train, y_pred_proba_train)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)
    '''
    get performace changing cutoff point
    : 선별된 데이터셋, 마커셋, 학습모델의 따른 결과의 다양한 cutoff에 따른 성능을 도출한다.

    y_train : getPC_df_othdisease에서 분류된 질병 클래스 중 훈련셋으로 분류된 부분
    y_pred_proba_train : train_evaluate에서 y_train을 가지고 predict_proba함수를 거쳐 도출된 확률의 집합
    y_test: getPC_df_othdisease에서 분류된 질병 클래스 중 테스트셋으로 분류된 부분
    y_pred_proba_test: y_test의 predict_proba 확률 집합
    min_senup: train_evaluate에서 getResults_changing_cutoff을 이용할 때 설정하는 sensitivity 값의 최소값 후보
    min_speup: train_evaluate에서 getResults_changing_cutoff을 이용할 때 설정하는 specificity 값의 최소값 후보
    min_senspe: sensitivity 와 specificity 의 최소값을 동시에 설정할 때의 후보
    max_youden: youden(true positive rate - false positive rate)값의 최소값 후보

    roc_curve에서 도출된 y_train, y_test의 tpr, fpr값의 집합을 통해
    sensitivity(tpr)과 specificity(1 - fpr), youden(tpr - fpr)을 구할 수 있다.
    이 중 조건에 부합하는 값 중 최적의 값의 인덱스를 찾아 그 때의 threshold를 cutoff로 한다.

    roc_auc_score : roc curve의 area under curve(모든 cutoff의 fpr, tpr point로 생기는 curve 아래의 넓이 값)
    '''
    ## 1.SEN__UP : 민감도 __ 이상일 때 특이도 최대값
    if len(min_senup) > 0:
        for min_sen in min_senup:
            min_sen_idx = np.where(sensitivity_test >= min_sen * 0.01)[0]
            spes = [specificity_test[i] for i in min_sen_idx]  # sen__ 이상의 spe
            try:
                max_spe_sidx = np.argmax(spes)  # 그중 max_spe의 min_sen_idx에서의 idx
                optimal_idx = min_sen_idx[max_spe_sidx]  # sen__ 이상 max_spe의 idx
                threshold = thresholds[optimal_idx]

                sen_test, spe_test = sensitivity_test[optimal_idx], specificity_test[optimal_idx]
                youd_test = youden_test[optimal_idx]

                y_pred_train = (y_pred_proba_train > threshold).astype(int)
                cm_train = confusion_matrix(y_train, y_pred_train, labels=[1, 0])
                tp_train, fn_train, fp_train, tn_train = cm_train.ravel()
                sen_train, spe_train = tp_train / (tp_train + fn_train), tn_train / (tn_train + fp_train)
                youd_train = sen_train + spe_train - 1

                train_perf = ['', '', '', 'train', sen_train, spe_train, auc_train, youd_train,
                              'SEN{}UP'.format(min_sen)]
                test_perf = ['', '', '', 'test', sen_test, spe_test, auc_test, youd_test, 'SEN{}UP'.format(min_sen)]
                results.append(train_perf)
                results.append(test_perf)
            except:
                pass

    '''
    !!
    getResults_changing_cutoff에서 sensitivity 후보값(min_senup)이 옵션으로 입력됐을 때 optimal threshold를 찾는다.

    y_test의 roc_curve에서 return된 tpr_test(sensitivity_test)가 최소값(현재 90 or 95) 이상일 때의 
    sensitivity_test 값의 인덱스를 np.where을 통해 min_sen index에 저장하고,
    specificity_test[min_sen index] : 같은 인덱스에 해당하는 specificity_test값들을 spes에 저장한다.

    np.argmax(spes)을 통해 그 중 가장 큰 specificity_test값을 가지는 인덱스를 optimal_idx로 하고 그 때의
    cutoff(thresholds[optimal_idx])를 threshold에 저장한다.

    y_train모델의 경우 앞서 y_test에서 결졍된 optimal threshold을 cutoff로 설정하여 
    y_pred_proba_train가 threshold보다 큰 경우를 y_pred_train에 저장하여 새로운 confusion matrix를 도출한다.
    cm_train.ravel()을 통해 confusion matrix(cm)의 결과(TP, FP, TN, FN)을 각각 저장하고,
    sensitivity_train과 specificity_train, youden_train을 도출할 수 있다.

    결과적으로 min_senup 후보 별 optimal threshold에서의 
    (sensitivity, specificity, auc, youden, 옵션명)의 값을 담은 train_perf/test_perf를
    results에 1행씩 append하는데 0~2열은 main의 마커셋, 마커의 수, 적용모델이다.
    '''

    ## 2.SPE__UP : 특이도 __ 이상일 때 민감도 최대값
    if len(min_speup) > 0:
        for min_spe in min_speup:
            min_spe_idx = np.where(specificity_test >= min_spe * 0.01)[0]
            sens = [sensitivity_test[i] for i in min_spe_idx]  # spe__ 이상의 sen
            try:
                max_sen_sidx = np.argmax(sens)  # 그중 max_sen의 min_spe_idx에서의 idx
                optimal_idx = min_spe_idx[max_sen_sidx]  # spe__ 이상 max_sen의 idx
                threshold = thresholds[optimal_idx]

                sen_test, spe_test = sensitivity_test[optimal_idx], specificity_test[optimal_idx]
                youd_test = youden_test[optimal_idx]

                y_pred_train = (y_pred_proba_train > threshold).astype(int)
                cm_train = confusion_matrix(y_train, y_pred_train, labels=[1, 0])
                tp_train, fn_train, fp_train, tn_train = cm_train.ravel()
                sen_train, spe_train = tp_train / (tp_train + fn_train), tn_train / (tn_train + fp_train)
                youd_train = sen_train + spe_train - 1

                train_perf = ['', '', '', 'train', sen_train, spe_train, auc_train, youd_train,
                              'SPE{}UP'.format(min_spe)]
                test_perf = ['', '', '', 'test', sen_test, spe_test, auc_test, youd_test, 'SPE{}UP'.format(min_spe)]
                results.append(train_perf)
                results.append(test_perf)
            except:
                pass
    '''
    min_senup이 주어졌을 때와 같은 방법으로
    getResults_changing_cutoff에서 specificity의 최소값 후보가 min_speup으로 주어졌을 때
    도출된 optimal threshold로부터의 train_perf/test_perf list를 results에 append한다.
    '''

    ## 3.SENSPE__UP : 민감도, 특이도 모두 __ 이상일 때 민감도 최대값


    '''
    getResults_changing_cutoff에서 max_youden(default=true)으로 설정된 경우의 처리 과정으로
    roc_curve에서 도출된 tpr, fpr을 통해
    youden(tpr - fpr)가 최대값일 때의 인덱스롤 np.argmax(youden_test)를 optimal_idx에 저장한다.
    
    min_senup, min_speup이 주어졌을 때와 같은 방법으로
    optimal_index의 threshold로부터의 train_perf/test_perf list를 results에 append한다.
    '''
    return results



def plot_roc(y, y_pred, legend=None, ax=None):
    roc = RocCurveDisplay.from_predictions(y, y_pred, name=legend, ax=ax)
    return roc
'''
sklearn의 RocCurveDisplay의 from_predictions함수는 실제 클래스 집합과 예측 클래스 집합에서 도출된 TPR, FPR의
roc curve 그래프에 대한 정보를 출력한다.
plot_roc함수는 이 정보를 roc에 저장하고 return한다.
'''