---
title: "6.2 项目：客户流失预测"
sidebar_position: 20
description: "完整的分类项目实战：不平衡数据处理、SMOTE、特征重要性分析、业务洞察"
keywords: [客户流失, 分类, 不平衡数据, SMOTE, 特征重要性, 业务洞察]
---

# 项目二：客户流失预测（分类问题）

:::tip 项目定位
客户流失预测是**最经典的商业 ML 应用**之一。本项目重点练习：不平衡数据处理、业务指标理解、从模型结果中提取业务洞察。
:::

## 项目概览

| 信息 | 说明 |
|------|------|
| 任务类型 | 二分类（流失/留存） |
| 核心挑战 | 数据不平衡（流失客户远少于留存） |
| 评估指标 | F1、AUC、召回率 |
| 涉及技能 | 不平衡处理、Pipeline、业务分析 |

---

## Step 1：模拟数据

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# 生成不平衡的客户数据
X, y = make_classification(
    n_samples=5000, n_features=15, n_informative=8,
    n_redundant=3, weights=[0.85, 0.15],  # 85% 留存, 15% 流失
    random_state=42
)

feature_names = ['月消费', '通话时长', '流量使用', '客服通话次数', '合同时长',
                 '账单争议', '套餐等级', '家庭成员数', '在网时长', '上月投诉',
                 '流量超限次数', '国际漫游', '增值服务数', '账户余额', '设备更换']

df = pd.DataFrame(X, columns=feature_names)
df['流失'] = y

print(f"数据形状: {df.shape}")
print(f"流失比例: {df['流失'].mean():.1%}")
print(f"流失客户: {df['流失'].sum()}, 留存客户: {(1-df['流失']).sum():.0f}")
```

---

## Step 2：不平衡数据处理

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

X = df.drop('流失', axis=1)
y = df['流失']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 方法1: 类别权重
rf_weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)
y_pred = rf_weighted.predict(X_test)

print("带类别权重的随机森林:")
print(classification_report(y_test, y_pred, target_names=['留存', '流失']))
print(f"AUC: {roc_auc_score(y_test, rf_weighted.predict_proba(X_test)[:,1]):.4f}")
```

### SMOTE 过采样

```python
# pip install imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote_pipe = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    smote_pipe.fit(X_train, y_train)
    y_pred_smote = smote_pipe.predict(X_test)

    print("\nSMOTE + 随机森林:")
    print(classification_report(y_test, y_pred_smote, target_names=['留存', '流失']))
except ImportError:
    print("请安装 imbalanced-learn: pip install imbalanced-learn")
```

---

## Step 3：特征重要性与业务洞察

```python
# 特征重要性
importance = rf_weighted.feature_importances_
sorted_idx = np.argsort(importance)

plt.figure(figsize=(8, 8))
plt.barh(range(len(sorted_idx)), importance[sorted_idx], color='coral')
plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
plt.xlabel('特征重要性')
plt.title('客户流失预测——特征重要性')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# 业务建议
print("\n业务洞察:")
top3 = np.array(feature_names)[np.argsort(importance)[-3:]]
for i, feat in enumerate(reversed(top3), 1):
    print(f"  {i}. {feat} 对流失预测最重要")
```

---

## Step 4：ROC 对比

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

models = {
    '逻辑回归': make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000)),
    '随机森林': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
}

plt.figure(figsize=(8, 6))
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('客户流失预测 ROC 对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 项目检查清单

- [ ] 分析数据不平衡程度
- [ ] 尝试至少 2 种不平衡处理方法（类别权重、SMOTE）
- [ ] 用 F1 和 AUC 评估（不只看准确率）
- [ ] 分析特征重要性，给出业务建议
- [ ] ROC 曲线多模型对比
