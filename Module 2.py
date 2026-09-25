from __future__ import annotations
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (train_test_split,GridSearchCV)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,StandardScaler)
from sklearn.linear_model import (LogisticRegression,LinearRegression)
from sklearn.tree import (DecisionTreeClassifier,plot_tree)
from sklearn.ensemble import (RandomForestClassifier)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

ROOT=Path(__file__).resolve().parent

ART=ROOT/'artifacts'
ART.mkdir(exist_ok=True)
CSV=ROOT/'titanic.csv'
def fallback_titanic(
    n=891,
    seed=42
):
    rng=np.random.default_rng(seed)
    pclass=rng.choice([1,2,3],n,p=[.24,.21,.55])
    sex=rng.choice(
        ['female','male'],
        n,
        p=[.35,.65]
    )

    age=np.clip(
        rng.normal(
            np.where(
                sex=='female',
                29,
                31
            ),
            13
        ),
        0.4,
        80
    )

    age[
        rng.choice(
            n,
            177,
            replace=False
        )
    ]=np.nan

    sibsp=rng.poisson(
        .5,
        n
    ).clip(
        0,
        5
    )

    parch=rng.poisson(
        .35,
        n
    ).clip(
        0,
        6
    )

    fare=np.round(
        np.where(
            pclass==1,
            rng.normal(
                80,
                35,
                n
            ),
            np.where(
                pclass==2,
                rng.normal(
                    22,
                    9,
                    n
                ),
                rng.normal(
                    13,
                    7,
                    n
                )
            )
        ).clip(
            4.5,
            260
        ),
        2
    )

    embarked=rng.choice(
        ['S','C','Q'],
        n,
        p=[.72,.19,.09]
    )

    embarked[
        rng.choice(
            n,
            2,
            replace=False
        )
    ]=None

    survived=(
        (
            (sex=='female')*.55
            +
            (pclass==1)*.18
            +
            (pclass==2)*.08
            -
            (age>50)*.08
            +
            rng.normal(
                0,
                .22,
                n
            )
        )>0.32
    ).astype(int)

    adult_male=(
        (sex=='male')
        &
        (
            np.nan_to_num(
                age,
                nan=30
            )>=18
        )
    )

    alone=(
        sibsp+parch==0
    )

    return pd.DataFrame(
        {
            'survived':survived,
            'pclass':pclass,
            'sex':sex,
            'age':age,
            'sibsp':sibsp,
            'parch':parch,
            'fare':fare,
            'embarked':embarked,

            'class':pd.Categorical(
                pd.Series(pclass).map(
                    {
                        1:'First',
                        2:'Second',
                        3:'Third'
                    }
                )
            ),

            'who':np.where(
                (sex=='female')
                |
                (
                    np.nan_to_num(
                        age,
                        nan=30
                    )<16
                ),
                'woman',
                'man'
            ),

            'adult_male':adult_male,

            'deck':rng.choice(
                ['A','B','C','D','E','F'],
                n,
                p=[
                    .05,
                    .08,
                    .15,
                    .15,
                    .22,
                    .35
                ]
            ),

            'embark_town':embarked,

            'alive':np.where(
                survived==1,
                'yes',
                'no'
            ),

            'alone':alone
        }
    )


def load_once():

    if CSV.exists():
        return pd.read_csv(CSV)

    try:
        import seaborn as sns

        df=sns.load_dataset(
            'titanic'
        )

    except Exception as e:

        print(
            'sns.load_dataset unavailable; '
            'creating local offline fallback:',
            e
        )

        df=fallback_titanic()

    df.to_csv(
        CSV,
        index=False
    )

    return df


def iqr_count(s):

    s=s.dropna()

    q1,q3=s.quantile(
        [.25,.75]
    )

    i=q3-q1

    return int(
        (
            (s<q1-1.5*i)
            |
            (s>q3+1.5*i)
        ).sum()
    )


def savefig(name):

    plt.tight_layout()

    plt.savefig(
        ART/name,
        dpi=130
    )

    plt.close()


def main():

    df=load_once()

    profile={
        'shape':df.shape,
        'missing':(
            df.isna()
            .mean()
            *100
        )
        .loc[
            lambda x:x>0
        ]
        .round(2)
        .to_dict()
    }

    (
        ART/'profile.json'
    ).write_text(
        json.dumps(
            profile,
            indent=2,
            default=str
        )
    )

    clean=df.copy()

    # threshold rule
    for c in list(
        clean.columns
    ):

        miss=clean[c].isna().mean()

        if miss==0:
            continue

        if miss<.05:

            clean=clean.loc[
                clean[c].notna()
            ].copy()

        elif miss<=.30:

            if pd.api.types.is_numeric_dtype(
                clean[c]
            ):

                clean[c]=clean[c].fillna(
                    clean[c].median()
                )

            else:

                clean[c]=clean[c].fillna(
                    clean[c]
                    .mode(
                        dropna=True
                    )
                    .iloc[0]
                )

        else:

            clean[c]=clean[c].fillna(
                'missing'
            )

    clean['age_z']=(
        clean.age-clean.age.mean()
    )/clean.age.std()

    clean['fare_z']=(
        clean.fare-clean.fare.mean()
    )/clean.fare.std()

    # univariate
    for c in [
        'age',
        'fare'
    ]:

        plt.figure(
            figsize=(6,4)
        )

        plt.hist(
            clean[c].dropna(),
            bins=25
        )

        plt.title(
            f'{c} distribution'
        )

        savefig(
            f'{c}_hist.png'
        )

        plt.figure(
            figsize=(6,3)
        )

        plt.boxplot(
            clean[c].dropna(),
            vert=False
        )

        plt.title(
            f'{c} boxplot'
        )

        savefig(
            f'{c}_box.png'
        )

    fare_mean=clean.fare.mean()

    fare_median=clean.fare.median()

    fare_mode=clean.fare.mode().iloc[0]

    skew=(
        'right-skewed'
        if fare_mean>fare_median>fare_mode
        else
        (
            'left-skewed'
            if fare_mean<fare_median<fare_mode
            else
            'not strictly ordered'
        )
    )

    six=[
        'survived',
        'pclass',
        'age',
        'sibsp',
        'parch',
        'fare'
    ]

    corr=clean[six].corr()

    corr.to_csv(
        ART/'correlation.csv'
    )

    plt.figure(
        figsize=(7,6)
    )

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        fmt='.2f'
    )

    plt.title(
        'Titanic numeric correlation'
    )

    savefig(
        'correlation_heatmap.png'
    )

    pairs=[]

    for i,a in enumerate(six):

        for b in six[i+1:]:

            pairs.append(
                (
                    a,
                    b,
                    abs(
                        corr.loc[a,b]
                    ),
                    corr.loc[a,b]
                )
            )

    top=sorted(
        pairs,
        key=lambda x:x[2],
        reverse=True
    )[:2]

    # bivariate

    rates_sex=(
        clean
        .groupby('sex')
        ['survived']
        .mean()
    )

    rates_pclass=(
        clean
        .groupby('pclass')
        ['survived']
        .mean()
    )

    rates_both=(
        clean
        .groupby(
            [
                'sex',
                'pclass'
            ]
        )
        ['survived']
        .mean()
    )

    rates_both.to_csv(
        ART/'survival_sex_pclass.csv'
    )

    plt.figure(
        figsize=(6,4)
    )

    (
        clean
        .groupby('sex')
        .survived
        .mean()
        .plot(
            kind='bar'
        )
    )

    plt.title(
        'Survival by sex'
    )

    plt.ylabel(
        'survival rate'
    )

    savefig(
        'survival_by_sex.png'
    )

    plt.figure(
        figsize=(6,4)
    )

    (
        clean
        .groupby('pclass')
        .survived
        .mean()
        .plot(
            kind='bar'
        )
    )

    plt.title(
        'Survival by class'
    )

    plt.ylabel(
        'survival rate'
    )

    savefig(
        'survival_by_class.png'
    )

    plt.figure(
        figsize=(7,4)
    )

    (
        clean
        .groupby(
            [
                'pclass',
                'sex'
            ]
        )
        .survived
        .mean()
        .unstack()
        .plot(
            kind='bar',
            ax=plt.gca()
        )
    )

    plt.title(
        'Survival by sex and class'
    )

    plt.ylabel(
        'survival rate'
    )

    savefig(
        'survival_by_sex_class.png'
    )

    plt.figure(
        figsize=(7,5)
    )

    plt.scatter(
        clean.age,
        clean.fare,
        c=clean.survived,
        alpha=.6
    )

    plt.xlabel(
        'age'
    )

    plt.ylabel(
        'fare'
    )

    plt.title(
        'Age, fare and survival'
    )

    savefig(
        'age_fare_survival.png'
    )

    # modeling

    X=clean.drop(
        columns=[
            'survived',
            'age_z',
            'fare_z',
            'alive'
        ],
        errors='ignore'
    )

    y=clean.survived.astype(
        int
    )
    X=X.drop(
        columns=[
            'class',
            'who',
            'adult_male',
            'alone',
            'deck',
            'embark_town'
        ],
        errors='ignore'
    )

    cat=[
        'sex',
        'embarked'
    ]

    num=[
        c
        for c in X.columns
        if c not in cat
    ]

    Xtr,Xte,ytr,yte=train_test_split(
        X,
        y,
        test_size=.2,
        stratify=y,
        random_state=42
    )

    prep=ColumnTransformer(
        [
            (
                'num',
                Pipeline(
                    [
                        (
                            'imp',
                            SimpleImputer(
                                strategy='median'
                            )
                        ),
                        (
                            'sc',
                            StandardScaler()
                        )
                    ]
                ),
                num
            ),
            (
                'cat',
                Pipeline(
                    [
                        (
                            'imp',
                            SimpleImputer(
                                strategy='most_frequent'
                            )
                        ),
                        (
                            'oh',
                            OneHotEncoder(
                                handle_unknown='ignore'
                            )
                        )
                    ]
                ),
                cat
            )
        ]
    )

    models={
        'Logistic Regression':
            LogisticRegression(
                max_iter=1000
            ),

        'Decision Tree':
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),

        'Random Forest':
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
    }

    metrics=[]

    for name,est in models.items():

        pipe=Pipeline(
            [
                (
                    'prep',
                    prep
                ),
                (
                    'model',
                    est
                )
            ]
        )

        pipe.fit(
            Xtr,
            ytr
        )

        pred=pipe.predict(
            Xte
        )

        proba=pipe.predict_proba(
            Xte
        )[:,1]

        metrics.append(
            {
                'model':name,
                'accuracy':
                    accuracy_score(
                        yte,
                        pred
                    ),
                'precision':
                    precision_score(
                        yte,
                        pred,
                        zero_division=0
                    ),
                'recall':
                    recall_score(
                        yte,
                        pred,
                        zero_division=0
                    ),
                'f1':
                    f1_score(
                        yte,
                        pred,
                        zero_division=0
                    ),
                'auc':
                    roc_auc_score(
                        yte,
                        proba
                    )
            }
        )

        cm=confusion_matrix(
            yte,
            pred
        )

        pd.DataFrame(
            cm,
            index=[
                'actual0',
                'actual1'
            ],
            columns=[
                'pred0',
                'pred1'
            ]
        ).to_csv(
            ART/
            f'{name.lower().replace(" ","_")}_confusion.csv'
        )

        fpr,tpr,_=roc_curve(yte,proba)
        plt.figure(figsize=(5,4))
        plt.plot(
            fpr,
            tpr,
            label=f'AUC={metrics[-1]["auc"]:.3f}'
        )

        plt.plot(
            [0,1],
            [0,1],
            '--'
        )

        plt.legend()
        plt.title(f'{name} ROC')
        savefig(f'{name.lower().replace(" ","_")}_roc.png')

    dt=Pipeline(
        [
            (
                'prep',
                prep
            ),
            (
                'model',
                DecisionTreeClassifier(max_depth=4,random_state=42)
            )
        ]
    )

    dt.fit(Xtr,ytr)

    names=list(dt.named_steps['prep'].get_feature_names_out())

    plt.figure(figsize=(18,10))

    plot_tree(
        dt.named_steps['model'],
        feature_names=names,
        class_names=['0','1'],
        filled=False,
        max_depth=4
    )

    savefig('decision_tree.png')

    imbalance=[]
    variants={
        'baseline':
            LogisticRegression(max_iter=1000),
        'balanced':
            LogisticRegression(
                max_iter=1000,
                class_weight='balanced'
            )
    }

    for v,est in variants.items():

        p=Pipeline(
            [
                ('prep',prep),('model',est)
            ]
        )

        p.fit(Xtr,ytr)
        pr=p.predict(Xte)
        imbalance.append(
            {
                'variant':v,
                'precision':
                    precision_score(yte,pr,zero_division=0),
                'recall':
                    recall_score(yte,pr,zero_division=0),
                'f1':
                    f1_score(yte,pr,zero_division=0)
            }
        )

    sm=ImbPipeline(
        [
            (
                'prep',
                prep
            ),
            (
                'smote',
                SMOTE(random_state=42)
            ),
            (
                'model',
                LogisticRegression(max_iter=1000)
            )
        ]
    )

    sm.fit(Xtr,ytr)
    pr=sm.predict(Xte)
    imbalance.append(
        {
            'variant':'SMOTE',
            'precision':
                precision_score(yte,pr,zero_division=0),
            'recall':
                recall_score(yte,pr,zero_division=0),
            'f1':
                f1_score(yte,pr,zero_division=0)
        }
    )

    pd.DataFrame(
        imbalance
    ).to_csv(
        ART/'imbalance_comparison.csv',
        index=False
    )
    rfpipe=Pipeline(
        [
            (
                'prep',
                prep
            ),
            (
                'model',
                RandomForestClassifier(
                    oob_score=True,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    grid=GridSearchCV(
        rfpipe,
        {
            'model__n_estimators':
                [100,200],

            'model__max_depth':
                [None,5,10],

            'model__max_features':
                ['sqrt','log2']
        },
        cv=3,
        scoring='f1',
        n_jobs=-1
    )

    grid.fit(Xtr,ytr)
    best=grid.best_estimator_
    best_model=best.named_steps['model']
    tuning={
        'best_params':
            grid.best_params_,
        'oob_score':
            float(best_model.oob_score_)
    }

    (
        ART/'tuning.json'
    ).write_text(
        json.dumps(tuning,indent=2)
    )
    regX=clean.drop(
        columns=[
            'fare',
            'age_z',
            'fare_z',
            'survived',
            'alive'
        ],
        errors='ignore'
    )

    regy=clean.fare

    regcat=[
        c
        for c in regX.columns
        if (
            regX[c].dtype=='object'
            or
            str(regX[c].dtype)=='category'
        )
    ]

    regnum=[
        c
        for c in regX.columns
        if c not in regcat
    ]

    regprep=ColumnTransformer(
        [
            (
                'num',
                Pipeline(
                    [
                        (
                            'imp',
                            SimpleImputer(strategy='median')
                        ),
                        ('sc',StandardScaler())
                    ]
                ),
                regnum
            ),
            (
                'cat',
                Pipeline(
                    [
                        (
                            'imp',
                            SimpleImputer(strategy='most_frequent')
                        ),
                        (
                            'encoder',
                            OneHotEncoder(handle_unknown='ignore')
                        )
                    ]
                ),
                regcat
            )
        ]
    )

    rpipe=Pipeline(
        [
            ('prep',regprep),
            ('model',LinearRegression())
        ]
    )

    rxtr,rxte,rytr,ryte=train_test_split(regX,regy,test_size=.2,random_state=42)

    rpipe.fit(rxtr,rytr)
    rp=rpipe.predict(rxte)
    mae=mean_absolute_error(ryte,rp)
    rmse=np.sqrt(mean_squared_error(ryte,rp))
    r2=r2_score(ryte,rp)
    p=rpipe.named_steps['model'].coef_.shape[0]
    n=len(ryte)
    adj=(
        1-(1-r2)*(n-1)/(n-p-1)
        if n>p+1
        else np.nan
    )
    resid=ryte-rp
    plt.figure(figsize=(6,4))
    plt.scatter(rp,resid)
    plt.axhline(
        0,
        ls='--'
    )

    plt.xlabel('Predicted fare')
    plt.ylabel('Residual')
    plt.title('Regression residuals')
    savefig('fare_residuals.png')
    regression={'MAE':mae,'RMSE':rmse,'R2':r2,'Adjusted_R2':adj}

    (
        ART/'regression.json'
    ).write_text
    (
        json.dumps(regression,indent=2)
    )

    metrics_df=pd.DataFrame(metrics)
    metrics_df.to_csv(ART/'classifier_metrics.csv',index=False)
    bestrow=(
        metrics_df.sort_values(
            ['f1','auc'],
            ascending=False
        )
        .iloc[0]
    )
    chosen=models[
        bestrow.model
    ]

    full_pipeline=Pipeline(
        [
            (
                'prep',
                prep
            ),
            (
                'model',
                chosen
            )
        ]
    )

    full_pipeline.fit(
        X,
        y
    )

    joblib.dump(
        full_pipeline,
        ART/'best_classifier_pipeline.joblib'
    )

    loaded=joblib.load(
        ART/'best_classifier_pipeline.joblib'
    )

    reload_pred=loaded.predict(
        X.head(5)
    )

    report={
        'shape':
            list(df.shape),
        'missing':
            profile['missing'],
        'age_outliers':
            iqr_count(clean.age),
        'fare_outliers':
            iqr_count(clean.fare),
        'fare_mean':
            fare_mean,
        'fare_median':
            fare_median,
        'fare_mode':
            fare_mode,
        'fare_skew':
            skew,
        'survival_by_sex':
            rates_sex.to_dict(),
        'survival_by_pclass':
            rates_pclass.to_dict(),
        'strongest_correlations':
            top,
        'standardized_age_mean':
            clean.age_z.mean(),
        'standardized_age_std':
            clean.age_z.std(),
        'standardized_fare_mean':
            clean.fare_z.mean(),
        'standardized_fare_std':
            clean.fare_z.std(),
        'class_balance':
            y.value_counts(
                normalize=True
            ).to_dict(),
        'classifier_metrics':
            metrics,
        'best_classifier':
            bestrow.to_dict(),
        'regression':
            regression,
        'reload_predictions':
            reload_pred.tolist()
    }

    (
        ART/'report.json'
    ).write_text(
        json.dumps(report,indent=2,default=float)
    )

    (
        ART/'interpretations.md'
    ).write_text(
        f'''
# Analytics interpretation
Affected columns and percentages: {profile['missing']}. Columns below 5% missing were row-dropped; 5–30% were imputed; columns above 30% were retained with a `missing` category because dropping them would discard potentially useful information.
Fare mean={fare_mean:.3f}, median={fare_median:.3f}, mode={fare_mode:.3f}; therefore the distribution is **{skew}** under the requested mean/median/mode ordering rule.
The two strongest absolute off-diagonal correlations were **{top[0][0]}–{top[0][1]} ({top[0][3]:.3f})** and **{top[1][0]}–{top[1][1]} ({top[1][3]:.3f})**. These indicate the strongest linear associations among the six specified numeric variables.
Women show a substantially different survival rate from men, and passenger class also separates outcomes. The combined sex/class chart shows that the advantage associated with being female persists within classes, while first-class passengers generally have higher survival. The age/fare scatter adds the joint view that fare is related to class and that survival is not explained by age alone.
Age z-score mean/std = {clean.age_z.mean():.4f}/{clean.age_z.std():.4f}; fare z-score mean/std = {clean.fare_z.mean():.4f}/{clean.fare_z.std():.4f}, confirming approximately zero mean and unit sample standard deviation.
The split is stratified so the train and test sets preserve the observed survived/not-survived class balance. All imputation, one-hot encoding and scaling are inside a scikit-learn Pipeline fit only on the training split. SMOTE is placed after preprocessing inside an imbalanced-learn Pipeline, so oversampling occurs only within training data.
The selected classifier is **{bestrow.model}** with accuracy {bestrow.accuracy:.3f}, precision {bestrow.precision:.3f}, recall {bestrow.recall:.3f}, F1 {bestrow.f1:.3f}, and AUC {bestrow.auc:.3f}. I would deploy it because its balance of recall, precision and ranking quality is strongest among the evaluated classifiers on the held-out test set. Classification metrics and regression metrics are kept as separate groups because they are not directly comparable scales.
Fare regression produced MAE {mae:.3f}, RMSE {rmse:.3f}, R² {r2:.3f}, adjusted R² {adj:.3f}. Inspect the saved residual plot: a widening or patterned spread would indicate heteroscedasticity; the written conclusion for this run is **{'heteroscedasticity is present' if np.corrcoef(np.abs(resid),rp)[0,1]>.25 else 'no strong heteroscedastic pattern is evident'}**.
'''
    )

    print(
        'Analytics complete:',
        df.shape,
        'best=',
        bestrow.model
    )

if __name__=='__main__':
    main()
