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

from warnings import simplefilter, filterwarnings
from sklearn.exceptions import ConvergenceWarning
from scipy.integrate import simps
filterwarnings("ignore", category=ConvergenceWarning)
filterwarnings("ignore")
from tqdm import tqdm
import datetime as dt

# 민감도(참 양성 비율) 계산 함수
def calculate_sensitivity(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)

# 특이도(참 음성 비율) 계산 함수
def calculate_specificity(true_negative, false_positive):
    return true_negative / (true_negative + false_positive)

# Youden 지수 계산 함수
def calculate_youden_index(sensitivity, specificity):
    return sensitivity + specificity - 1


# 모델 train 함수
def train_evaluate(marker, model, dataset, gridsearchcv=True, model_param=None, save_plot_as=None, kfoldcv=False,
                   ax_train=None, ax_test=None):
    # 데이터 준비
    df = dataset

    X = df.loc[:, marker]
    y = df.loc[:, 'label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    X_train_without = X_train
    X_test_without = X_test
    
    sen_score = make_scorer(recall_score, pos_label=1)
    spe_score = make_scorer(recall_score, pos_label=0)

    # 그리드 서치를 사용하여 최적의 매개변수 탐색
    if gridsearchcv == True:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
        grid_model = GridSearchCV(estimator=model,
                                  param_grid=model_param,
                                  cv=cv,
                                  scoring='roc_auc',
                                  n_jobs=-1,
                                  verbose=2)
        grid_model.fit(X_train, y_train)
        clf = grid_model.best_estimator_
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
    else:
        clf = model
        clf.fit(X_train_without, y_train)
        
    # 모델 학습
    print(clf)
    clf.fit(X_train, y_train)

    
    if kfoldcv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        fig, ax = plt.subplots()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
    
        # Stratified K-Fold Cross Validation 수행
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            model.fit(X[train_idx], y[train_idx])
            y_pred_proba = model.predict_proba(X[test_idx])[:, 1]
            fpr, tpr, _ = roc_curve(y[test_idx], y_pred_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(auc(fpr, tpr))
    
        # 평균 ROC 곡선 계산
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        
        # 평균 ROC 곡선 그리기
        ax.plot(mean_fpr, mean_tpr, color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2, alpha=0.8
                )
        ax.legend(loc="lower right")
        plt.show()
        
       # Cross Validation을 사용하여 성능 지표 계산
        spe_score = make_scorer(recall_score, pos_label=0)
        sen_score = make_scorer(recall_score, pos_label=1)
        cv_result = cross_validate(model, X, y, cv=5,
                               scoring={'sen': sen_score, 'spe': spe_score, 'auc': 'roc_auc'})

        # Cross Validation 결과를 기반으로 평균 성능 지표 계산
        sen_list = cv_result['test_sen']
        spe_list = cv_result['test_spe']
        auc_cv = mean_auc
        youden_list = sen_list + spe_list - 1
        youden_cv = np.mean(youden_list)
        spe_cv = np.mean(spe_list)
        sen_cv = np.mean(sen_list)


    # 테스트 데이터에 대한 예측 확률 계산
    y_pred_proba_test = clf.predict_proba(X_test_without)[:, 1]
    y_pred_proba_train = clf.predict_proba(X_train_without)[:, 1]


    # 만약 ax_train과 ax_test가 None이 아니라면 ROC 곡선을 그리고 반환합
    if ax_train != None and ax_test != None:
        legend = save_plot_as + '\n' + str(model).split('(')[0]  # best_auc_model 등 그릴때 원래 이 코드

        # 학습 데이터의 ROC 곡선 그리기
        train_roc = plot_roc(y_train, y_pred_proba_train, legend=legend, ax=ax_train)
        # 테스트 데이터의 ROC 곡선 그리기
        test_roc = plot_roc(y_test, y_pred_proba_test, legend=legend, ax=ax_test)

        return train_roc, test_roc
    
    # ax_train과 ax_test가 None이라면 임계값을 변경하여 성능 지표 계산
    else:
        results_list = getResults_changing_cutoff(y_train, y_pred_proba_train,
                                                  y_test, y_pred_proba_test,
                                                  min_senup=[90, 95], min_speup=[90, 95])

        # 만약 save_plot_as가 None이 아니라면 ROC 곡선을 그리고 그래프 저장
        if save_plot_as != None:
            title = ' '.join(marker) + '\n' + str(model).split('(')[0]
            if len(X.columns) > 7:
                title = ' '.join(marker[:7]) + '\n' + ' '.join(marker[7:]) + '\n' + str(model).split('(')[0]

            fig, ax = plt.subplots()

            train_roc = plot_roc(y_train, y_pred_proba_train, legend='train', ax=ax)
            test_roc = plot_roc(y_test, y_pred_proba_test, legend='test', ax=ax)
            plt.title(title)
            plt.savefig(save_plot_as + '/' + ' '.join(marker) + '_' + str(model).split('(')[0] + '.png')
        
        # Evaluation Metrics
        y_pred_test = clf.predict(X_test_without)
        y_pred_train = clf.predict(X_train_without)
        
        # Accuracy
        accuracy_train = clf.score(X_train_without, y_train)
        accuracy_test = clf.score(X_test_without, y_test)
        
        # Sensitivity (Recall)
        true_positive_test = np.sum((y_test == 1) & (y_pred_test == 1))
        false_negative_test = np.sum((y_test == 1) & (y_pred_test == 0))
        sensitivity_test = calculate_sensitivity(true_positive_test, false_negative_test)
        
        true_positive_train = np.sum((y_train == 1) & (y_pred_train == 1))
        false_negative_train = np.sum((y_train == 1) & (y_pred_train == 0))
        sensitivity_train = calculate_sensitivity(true_positive_train, false_negative_train)
        
        # Specificity
        true_negative_test = np.sum((y_test == 0) & (y_pred_test == 0))
        false_positive_test = np.sum((y_test == 0) & (y_pred_test == 1))
        specificity_test = calculate_specificity(true_negative_test, false_positive_test)
        
        true_negative_train = np.sum((y_train == 0) & (y_pred_train == 0))
        false_positive_train = np.sum((y_train == 0) & (y_pred_train == 1))
        specificity_train = calculate_specificity(true_negative_train, false_positive_train)
        
        # Youden's Index
        youden_index_test = calculate_youden_index(sensitivity_test, specificity_test)
        youden_index_train = calculate_youden_index(sensitivity_train, specificity_train)
        
        # ROC AUC Score
        roc_auc_score_test = roc_auc_score(y_test, y_pred_proba_test)
        roc_auc_score_train = roc_auc_score(y_train, y_pred_proba_train)
        
        # Confusion Matrix
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
        confusion_matrix_train = confusion_matrix(y_train, y_pred_train)
        
        # Classification Report
        classification_report_test = classification_report(y_test, y_pred_test)
        classification_report_train = classification_report(y_train, y_pred_train)
        
        # Printing the evaluation metrics
        print("Evaluation Metrics:")
        print("-------------------")
        print("Train Accuracy:", accuracy_train)
        print("Test Accuracy:", accuracy_test)
        print("Train Sensitivity (Recall):", sensitivity_train)
        print("Test Sensitivity (Recall):", sensitivity_test)
        print("Train Specificity:", specificity_train)
        print("Test Specificity:", specificity_test)
        print("Train Youden's Index:", youden_index_train)
        print("Test Youden's Index:", youden_index_test)
        print("Train ROC AUC Score:", roc_auc_score_train)
        print("Test ROC AUC Score:", roc_auc_score_test)
        print("Train Confusion Matrix:")
        print(confusion_matrix_train)
        print("Test Confusion Matrix:")
        print(confusion_matrix_test)
        print("Train Classification Report:")
        print(classification_report_train)
        print("Test Classification Report:")
        print(classification_report_test)   
                    
        # 결과 리스트와 모델 파라미터를 반환합니다.
        return results_list, model_param  


# 주어진 임계값에 따라 성능 지표 계산 및 결과 반환 함수
def getResults_changing_cutoff(y_train, y_pred_proba_train, y_test, y_pred_proba_test,
                               min_senup=[], min_speup=[], min_senspe=[]):
    results = []

    # 학습 데이터와 테스트 데이터에 대한 FPR, TPR, 임계값 계산
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
    fpr_test, tpr_test, thresholds = roc_curve(y_test, y_pred_proba_test)

    # 학습 데이터와 테스트 데이터에 대한 민감도, 특이도, Youden 지수 계산
    sensitivity_train, specificity_train, youden_train = tpr_train, 1 - fpr_train, tpr_train - fpr_train
    sensitivity_test, specificity_test, youden_test = tpr_test, 1 - fpr_test, tpr_test - fpr_test
   # 학습 데이터와 테스트 데이터에 대한 AUC 계산
    auc_train = roc_auc_score(y_train, y_pred_proba_train)
    auc_test = roc_auc_score(y_test, y_pred_proba_test)

     # min_senup 리스트에 주어진 임계값에 따라 민감도 상승 검사
    if len(min_senup) > 0:
        for min_sen in min_senup:
            # 주어진 임계값 이상의 민감도를 가진 인덱스 추출
            min_sen_idx = np.where(sensitivity_test >= min_sen * 0.01)[0]
            spes = [specificity_test[i] for i in min_sen_idx]  
            # 임계값 이상의 민감도에 해당하는 특이도 추출
            try:
                max_spe_sidx = np.argmax(spes) # 특이도 최대값의 인덱스
                optimal_idx = min_sen_idx[max_spe_sidx]  # 최적 인덱스 선택
                threshold = thresholds[optimal_idx] # 최적 임계값 선택

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


    return results


# ROC 곡선 그리기 함수
def plot_roc(y, y_pred, legend=None, ax=None):
    roc = RocCurveDisplay.from_predictions(y, y_pred, name=legend, ax=ax)
    return roc
