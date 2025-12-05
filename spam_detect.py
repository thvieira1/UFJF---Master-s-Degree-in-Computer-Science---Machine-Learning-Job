import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import LabelEncoder
import warnings
import time
import os

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
K_FOLDS = 5



def carregar_datasets():
    arquivos = [f for f in os.listdir('.') if f.endswith('.csv')]
    datasets = []
    
    print(f"Arquivos CSV encontrados: {arquivos}")
    
    for arquivo in arquivos:
        if 'resultado' in arquivo.lower():
            continue
            
        try:
            print(f"Tentando carregar: {arquivo}")
            df = pd.read_csv(arquivo, encoding='utf-8')
            
            if len(df) < 100:
                continue
            
            print(f"  Colunas: {df.columns.tolist()}")
            
            label_cols = [c for c in df.columns if 'label' in c.lower() or 'spam' in c.lower() or 'category' in c.lower()]
            text_cols = [c for c in df.columns if 'text' in c.lower() or 'message' in c.lower() or 'email' in c.lower() or 'body' in c.lower()]
            
            if not label_cols or not text_cols:
                continue
            
            label_col = label_cols[0]
            text_col = text_cols[0]
            
            df = df[[label_col, text_col]].copy()
            df.columns = ['label', 'message']
            
            if df['label'].dtype in [int, np.int64, np.int32]:
                df['label'] = df['label'].map({1: 'spam', 0: 'ham'})
            else:
                df['label'] = df['label'].str.lower().str.strip()
                label_map = {'1': 'spam', '0': 'ham', 'spam': 'spam', 'ham': 'ham', 'not spam': 'ham'}
                df['label'] = df['label'].map(label_map)
            
            df = df.dropna()
            
            if len(df) > 100 and df['label'].nunique() == 2:
                datasets.append(df)
                print(f"  ✓ Dataset carregado: {len(df)} amostras")
                
            if len(datasets) == 2:
                break
                
        except Exception as e:
            print(f"  ✗ Erro: {e}")
            continue
    
    if len(datasets) < 2:
        print("\nERRO: Não foram encontrados 2 datasets válidos!")
        exit(1)
    
    return datasets[0], datasets[1]


def processar_e_avaliar(df, dataset_name):
    X = df['message']
    y = df['label']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    X_tfidf = vectorizer.fit_transform(X)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=SEED, max_iter=1000, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=SEED, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=SEED, n_estimators=100, n_jobs=-1),
        'SVM': SVC(kernel='linear', random_state=SEED, probability=True),
        'Naive Bayes': MultinomialNB(),
        'XGBoost': XGBClassifier(random_state=SEED, eval_metric='logloss', use_label_encoder=False, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=SEED, n_estimators=100)
    }
    
    ensemble_voting = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=SEED, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=SEED, n_estimators=50, n_jobs=-1)),
            ('nb', MultinomialNB())
        ],
        voting='soft'
    )
    
    ensemble_weighted = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=SEED, max_iter=1000)),
            ('rf', RandomForestClassifier(random_state=SEED, n_estimators=50, n_jobs=-1)),
            ('nb', MultinomialNB())
        ],
        voting='soft',
        weights=[2, 3, 1]
    )
    
    models['Ensemble (Voting)'] = ensemble_voting
    models['Ensemble (Weighted)'] = ensemble_weighted
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'f1': make_scorer(f1_score, average='binary', zero_division=0),
        'roc_auc': 'roc_auc'
    }
    
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    results = {}
    
    for name, model in models.items():
        start_time = time.time()
        cv = cross_validate(model, X_tfidf, y_encoded, cv=skf, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - start_time
        
        results[name] = {
            'accuracy': cv['test_accuracy'].mean(),
            'precision': cv['test_precision'].mean(),
            'recall': cv['test_recall'].mean(),
            'f1': cv['test_f1'].mean(),
            'roc_auc': cv['test_roc_auc'].mean(),
            'time': elapsed
        }
    
    return pd.DataFrame(results).T.round(4), X, y_encoded


def criar_visualizacoes(results_ds1, results_ds2):
    sns.set_style("whitegrid")

    fig1, ax = plt.subplots(figsize=(12, 6))

    data = pd.DataFrame({
        'Dataset 1': results_ds1['f1'],
        'Dataset 2': results_ds2['f1']
    }).sort_values('Dataset 1', ascending=False)

    x = np.arange(len(data))
    width = 0.35

    ax.bar(x - width/2, data['Dataset 1'], width, label='Dataset 1', edgecolor='black')
    ax.bar(x + width/2, data['Dataset 2'], width, label='Dataset 2', edgecolor='black')

    ax.set_ylabel('F1-Score', fontweight='bold')
    ax.set_title('Comparação de F1-Score entre Modelos e Datasets', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('visao_geral_modelos.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Top 5 ---
    top5_ds1 = results_ds1.nlargest(5, 'f1')
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top5_ds1)))
    axes[0].barh(range(len(top5_ds1)), top5_ds1['f1'], color=colors, edgecolor='black')
    axes[0].set_yticks(range(len(top5_ds1)))
    axes[0].set_yticklabels(top5_ds1.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title('Top 5 Modelos – Dataset 1')

    for i, v in enumerate(top5_ds1['f1']):
        axes[0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8)

    axes[1].scatter(results_ds1['time'], results_ds1['f1'], s=120,
                    c=range(len(results_ds1)), cmap='viridis', edgecolor='black', alpha=0.8)
    for name, row in results_ds1.iterrows():
        axes[1].annotate(name, (row['time'], row['f1']), fontsize=7, ha='left', va='bottom')
    axes[1].set_xlabel('Tempo total de CV (s)', fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontweight='bold')
    axes[1].set_title('Trade-off Tempo vs Performance – Dataset 1')
    axes[1].grid(alpha=0.3)

    metrics_ds1 = results_ds1[['accuracy', 'precision', 'recall', 'f1']].copy()
    metrics_ds1 = metrics_ds1.sort_values('f1', ascending=False)

    idx = np.arange(len(metrics_ds1))
    width = 0.18

    axes[2].bar(idx - 1.5*width, metrics_ds1['accuracy'], width, label='Accuracy')
    axes[2].bar(idx - 0.5*width, metrics_ds1['precision'], width, label='Precision')
    axes[2].bar(idx + 0.5*width, metrics_ds1['recall'], width, label='Recall')
    axes[2].bar(idx + 1.5*width, metrics_ds1['f1'], width, label='F1-Score')

    axes[2].set_xticks(idx)
    axes[2].set_xticklabels(metrics_ds1.index, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylim(0.7, 1.0)
    axes[2].set_title('Métricas por Modelo – Dataset 1')
    axes[2].legend(fontsize=7)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('detalhe_dataset1.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    fig3, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    top5_ds2 = results_ds2.nlargest(5, 'f1')
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top5_ds2)))
    axes[0].barh(range(len(top5_ds2)), top5_ds2['f1'], color=colors, edgecolor='black')
    axes[0].set_yticks(range(len(top5_ds2)))
    axes[0].set_yticklabels(top5_ds2.index)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title('Top 5 Modelos – Dataset 2')

    for i, v in enumerate(top5_ds2['f1']):
        axes[0].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=8)

    axes[1].scatter(results_ds2['time'], results_ds2['f1'], s=120,
                    c=range(len(results_ds2)), cmap='plasma', edgecolor='black', alpha=0.8)
    for name, row in results_ds2.iterrows():
        axes[1].annotate(name, (row['time'], row['f1']), fontsize=7, ha='left', va='bottom')
    axes[1].set_xlabel('Tempo total de CV (s)', fontweight='bold')
    axes[1].set_ylabel('F1-Score', fontweight='bold')
    axes[1].set_title('Trade-off Tempo vs Performance – Dataset 2')
    axes[1].grid(alpha=0.3)

    metrics_ds2 = results_ds2[['accuracy', 'precision', 'recall', 'f1']].copy()
    metrics_ds2 = metrics_ds2.sort_values('f1', ascending=False)

    idx = np.arange(len(metrics_ds2))
    width = 0.18

    axes[2].bar(idx - 1.5*width, metrics_ds2['accuracy'], width, label='Accuracy')
    axes[2].bar(idx - 0.5*width, metrics_ds2['precision'], width, label='Precision')
    axes[2].bar(idx + 0.5*width, metrics_ds2['recall'], width, label='Recall')
    axes[2].bar(idx + 1.5*width, metrics_ds2['f1'], width, label='F1-Score')

    axes[2].set_xticks(idx)
    axes[2].set_xticklabels(metrics_ds2.index, rotation=45, ha='right', fontsize=8)
    axes[2].set_ylim(0.7, 1.0)
    axes[2].set_title('Métricas por Modelo – Dataset 2')
    axes[2].legend(fontsize=7)
    axes[2].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('detalhe_dataset2.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)


df1, df2 = carregar_datasets()

results_ds1, X1, y1 = processar_e_avaliar(df1, "Dataset 1")
results_ds2, X2, y2 = processar_e_avaliar(df2, "Dataset 2")

results_ds1.to_csv('resultados_dataset1_final.csv')
results_ds2.to_csv('resultados_dataset2_final.csv')

criar_visualizacoes(results_ds1, results_ds2)

print("\n" + "="*80)
print("RESULTADOS FINAIS")
print("="*80)
print("\nDataset 1 - Top 5:")
print(results_ds1.nlargest(5, 'f1')[['f1', 'accuracy', 'time']])
print("\nDataset 2 - Top 5:")
print(results_ds2.nlargest(5, 'f1')[['f1', 'accuracy', 'time']])
print("\n" + "="*80)
