---
title: "5.2.5 集成学习：森林、Boosting、Stacking"
description: "一节紧凑的集成学习实操课：对比单棵树、随机森林、梯度提升和防泄漏 Stacking。"
sidebar:
  order: 6
head:
  - tag: meta
    attrs:
      name: keywords
      content: "集成学习, Random Forest, Bagging, Boosting, GBDT, Stacking, XGBoost, LightGBM, CatBoost"
---
![Bagging 与 Boosting 对比图](/img/course/ch05-ensemble-bagging-boosting-flow.webp)

集成学习会把多个模型组合起来，减少单个模型弱点主导结果的风险。对表格数据来说，它经常是经典机器学习里最强的一类方法。

## 集成学习为什么从单棵树后面出现

单棵决策树很直观，但它有一个历史老问题：数据稍微换一批，树结构可能就大变。也就是说，单树方差高、容易过拟合。

集成学习的核心回答是：

```text
不要相信一棵树，让很多棵树一起投票或逐步纠错。
```

| 单树问题 | 集成路线 | 怎么解决 |
|---|---|---|
| 结构不稳定 | Random Forest / Bagging | 多棵树并行训练，投票或平均 |
| 单棵弱模型能力有限 | Boosting / GBDT | 后面的模型重点修正前面的错误 |
| 不同模型各有强项 | Stacking | 让二层模型学习如何组合基模型 |

XGBoost、LightGBM、CatBoost 这类工具之所以在表格任务里常见，是因为它们把 Boosting 做得更快、更稳、更工程化。但第一步仍然是先理解：它们是在解决“单模型不稳定或不够强”的问题。

## 先看两条主线

![集成学习家族漫画](/img/course/ch05-ensemble-family-comic.webp)

不要一上来背模型名，先分清三条主线：

- **Bagging**，例如 Random Forest：多个模型并行训练后投票。适合追求稳定、降低方差。要注意模型可能变大，解释性会下降。
- **Boosting**，例如 GBDT、XGBoost、LightGBM、CatBoost：后一个模型专门补前一个模型的错。适合追求表格数据精度。要用深度、学习率和 early stopping 控制过拟合。
- **Stacking**，例如 `StackingClassifier`：把基模型预测再喂给元模型。适合组合不同模型家族。构造时要用交叉验证，避免验证信息泄漏。

## 跑模型对比实验

新建 `ch05_ensemble_lab.py`。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.25,
    random_state=42,
    stratify=data.target,
)

models = {
    "single_tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    ),
    "gradient_boost": GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

models["stacking_cv"] = StackingClassifier(
    estimators=[
        ("rf", models["random_forest"]),
        ("gb", models["gradient_boost"]),
        ("lr", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42),
        )),
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    cv=5,
)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name:<15} accuracy={accuracy_score(y_test, pred):.3f} f1={f1_score(y_test, pred):.3f}")

rf = models["random_forest"]
importances = rf.feature_importances_
top = importances.argsort()[-3:][::-1]
print("top_rf_features=")
for idx in top:
    print(f"- {data.feature_names[idx]}: {importances[idx]:.3f}")
```

运行：

```bash
python ch05_ensemble_lab.py
```

预期输出：

```text
single_tree     accuracy=0.944 f1=0.956
random_forest   accuracy=0.958 f1=0.967
gradient_boost  accuracy=0.944 f1=0.956
stacking_cv     accuracy=0.986 f1=0.989
top_rf_features=
- worst perimeter: 0.146
- worst area: 0.140
- worst concave points: 0.109
```

![集成学习实验结果图](/img/course/ch05-ensemble-comparison-result-map.webp)

不同 sklearn 版本分数可能略有变化。保留对比表和重要特征，作为项目证据。

## 读懂结果

单棵树是 baseline。随机森林通过平均很多棵不同的树，通常会更稳定。

Boosting 不会在每个小数据集里自动更好。它需要控制树深、学习率、树数量和验证集表现。

Stacking 这里可能赢，是因为它组合了不同模型家族；但它必须用交叉验证。元模型如果直接看到基模型在训练行上的预测，就会发生信息泄漏。

## Bagging：随机森林

![集成学习投票与森林图](/img/course/ensemble-learning-voting-forest.webp)

随机森林会在随机化的数据视角上训练很多决策树，再对它们的预测做平均或投票。

优先记这些设置：

| 参数 | 控制什么 | 新手默认 |
|---|---|---|
| `n_estimators` | 树的数量 | `100` 到 `300` |
| `max_depth` | 树深 | 先小后大 |
| `min_samples_leaf` | 叶子最少样本数 | 过拟合时调大 |
| `random_state` | 可复现 | 学习时总是设置 |

## Boosting：GBDT 与工具链

![GBDT 残差修正漫画](/img/course/ch05-ensemble-gbdt-residual-correction.webp)

Boosting 是顺序建模：

```text
第一棵小树 -> 找错误 -> 下一棵小树关注错误 -> 重复
```

在 sklearn 里，先从 `GradientBoostingClassifier` 或 `HistGradientBoostingClassifier` 开始。真实表格项目里，XGBoost、LightGBM、CatBoost 很常见，但不要在 sklearn baseline 还没清楚前就直接加外部库。

![Boosting 工具选择漫画](/img/course/ch05-ensemble-boosting-toolkit.webp)

Boosting 第一次调参顺序：

| 步骤 | 调什么 | 为什么 |
|---|---|---|
| 1 | `learning_rate` 和 `n_estimators` | 控制步长和训练轮数 |
| 2 | `max_depth` / 叶子设置 | 控制复杂度 |
| 3 | 验证集或 early stopping | 防止过拟合 |
| 4 | 特征预处理 | 提高信号质量 |

## 安全使用 Stacking

![防泄漏 Stacking 工作流漫画](/img/course/ch05-ensemble-stacking-leakage-safe.webp)

Stacking 只有在元模型看到 out-of-fold 预测时才可靠：

```text
在 CV 折里训练基模型 -> 收集 out-of-fold 预测 -> 训练元模型 -> 在 holdout test 上评估
```

优先用 sklearn 的 `StackingClassifier(cv=5)`，不要手动把训练行上的预测直接喂给元模型。

## 怎么选模型

| 场景 | 先用 |
|---|---|
| 需要强而稳的 baseline | Random Forest |
| 表格数据有很多非线性 | Gradient Boosting / XGBoost / LightGBM |
| 类别特征很多 | CatBoost，放在 baseline 之后 |
| 不同模型家族表现互补 | 带交叉验证的 Stacking |
| 需要最容易解释 | 浅树或随机森林特征重要性 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
任务：带目标定义的回归或分类问题
模型：线性/逻辑回归/树/集成/SVM 配置和训练/测试划分
指标：回归误差、准确率/F1、阈值曲线或混淆矩阵
失败检查：过拟合、欠拟合、特征缩放、阈值选择或类别不平衡
期望产出：模型结果加错误样本或残差复查
```

## 常见错误

| 现象 | 先检查 | 常见修复 |
|---|---|---|
| 集成模型几乎不比单树好 | 特征弱或划分不稳定 | 加特征，用交叉验证 |
| 训练好、测试差 | 过拟合 | 降低深度、增大叶子样本、加验证 |
| Boosting 树越多越差 | 轮数太多 | 降低学习率或 early stopping |
| Stacking 分数异常完美 | 信息泄漏 | 用 out-of-fold 预测或 `StackingClassifier(cv=...)` |
| 过度解读特征重要性 | 特征相关性强 | 用 permutation importance 或消融验证 |

## 练习

1. 把随机森林 `max_depth` 从 `6` 改成 `3` 和 `None`。
2. 把 Gradient Boosting 的 `learning_rate` 从 `0.05` 改成 `0.2`。
3. 解释如果 Stacking 不用交叉验证，为什么会泄漏。
4. 保存模型对比表，并写一段话说明你会先上线哪个模型。

<details>
<summary>参考实现与讲解</summary>

1. `max_depth=3` 会限制单棵树复杂度，可能更稳但会欠拟合；`None` 允许树长得很深，训练分数可能更高，但测试分数和稳定性要重点检查。
2. `learning_rate=0.2` 会让每轮 Boosting 修正更大，可能更快提升，也可能更快过拟合。需要结合验证集或交叉验证判断。
3. Stacking 如果让二层模型看到“基模型在训练自己时给出的预测”，二层模型就会学到过于乐观的信号。用 out-of-fold 预测或 `StackingClassifier(cv=...)` 才能降低泄漏。
4. 上线选择不只看最高分，还要看测试集稳定性、训练/预测成本、可解释性和失败代价。写结论时要说明为什么选择某个模型，而不是只贴一张分数表。

</details>

## 通关检查

能解释下面五件事，就可以继续：

- Bagging 和 Boosting 有什么区别；
- 为什么随机森林通常比单棵树更稳；
- 为什么 Boosting 需要验证集控制；
- 为什么 Stacking 必须用交叉验证；
- 为什么排行榜最高分不一定是最适合生产的选择。
