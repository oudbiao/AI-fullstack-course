---
title: "ブランチと協働"
sidebar_position: 4
description: "ブランチを使って新機能を安全に開発し、Pull Request の流れを理解する"
---

# ブランチと協働

![Git ブランチ協働フローチャート](/img/course/ch01-git-branch-collaboration.png)

## この節の位置づけ

この節では、Git がどうして安全な共同作業を支えられるのかを説明します。ブランチを使うと、メインのコードを壊さずに新機能を試せることを理解し、Pull Request とマージコンフリクトの基本も学びます。これからチーム開発やオープンソースへの貢献を始めるための準備になります。

## 学習目標

- ブランチの概念と使いどころを理解する
- ブランチの作成、切り替え、マージの操作を身につける
- Pull Request の協働フローを理解する
- 簡単なマージコンフリクトを解決できるようになる

---

## ブランチとは？

### リフォームにたとえると

あなたがマンションの部屋に住んでいるとします（`main` ブランチ = 今住んでいる家）。新しい内装スタイルを試してみたいけれど、うまくいくかはまだ分かりません。

選択肢は 2 つあります。

1. **いきなり今の家を改造する** — 失敗したら住めなくなるかもしれません
2. **同じ間取りの部屋をもう1つ借りて（新しいブランチ）、そこで試す** — 気に入ったら戻す、微妙なら解約する

ブランチは 2 の方法です。新しいブランチで自由に変更し、うまくいったら `main` にマージ、失敗したらブランチを消すだけ。`main` には影響しません。

### コードでの実際の場面

```
あなたは AI 画像分類プロジェクトを進めています。main ブランチには、正常に動くコードがあります。

今やりたいこと:
  - モデルを CNN から Vision Transformer に変えたい
  - 効果が良くなるかはまだ分からない
  - 変更が大きく、数日かかるかもしれない

もし main で直接作業すると:
  ❌ 途中でコードが動かなくなるかもしれない
  ❌ 上司から突然 bug 修正を頼まれても、main がぐちゃぐちゃ
  ❌ 最後に ViT が微妙だと分かっても、50 ファイルも変えてしまった後では戻しにくい

ブランチを使えば:
  ✅ feature/vit ブランチで少しずつ変更できる
  ✅ bug 修正が来たら main に切り替えて対応し、また戻って続けられる
  ✅ ViT が合わなければ、ブランチを消すだけで main は無傷
```

---

## ブランチの基本操作

### ブランチを表示する

```bash
# ローカルブランチを表示する（現在のブランチには * が付く）
git branch
# 出力:
# * main

# すべてのブランチを表示する（リモートも含む）
git branch -a
```

### ブランチを作成して切り替える

```bash
# 新しいブランチを作成する
git branch feature/data-augmentation

# 新しいブランチへ切り替える
git checkout feature/data-augmentation

# あるいは一度で完了する方法（こちらがよく使われる）
git checkout -b feature/data-augmentation
```

:::tip ブランチ名の付け方
よく使われる名前の例：
- `feature/xxx` — 新機能（例: `feature/add-resnet`）
- `fix/xxx` — bug 修正（例: `fix/training-crash`）
- `experiment/xxx` — 試験的な取り組み（例: `experiment/try-vit`）
:::

### 例：ブランチで新機能を開発する

実際に操作してみましょう。前に使った `ai-image-classifier` プロジェクトを続けて使います。

```bash
cd ai-image-classifier

# 今が main ブランチか確認する
git branch
# * main

# 新しいブランチを作成して切り替える：データ拡張機能を追加する
git checkout -b feature/data-augmentation
```

これで新しいブランチに移動しました。コードを書いていきます。

```bash
# データ拡張モジュールを作成する
cat > src/augmentation.py << 'EOF'
import torchvision.transforms as T

def get_train_transforms():
    """訓練データの拡張方法"""
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),        # 50% の確率で左右反転
        T.RandomRotation(degrees=15),          # ±15 度のランダム回転
        T.ColorJitter(                         # 色の揺らぎ
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

def get_test_transforms():
    """テストデータは標準化のみで、拡張はしない"""
    return T.Compose([
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
EOF

# train.py を更新して、データ拡張を使う
cat >> src/train.py << 'EOF'

# 追加: データ拡張を使う
from augmentation import get_train_transforms, get_test_transforms
train_transform = get_train_transforms()
test_transform = get_test_transforms()
print("データ拡張の設定を読み込みました")
EOF

# 現在のブランチにコミットする
git add .
git commit -m "feat: データ拡張モジュールを追加（左右反転、回転、色の揺らぎ）"
```

今の 2 つのブランチの状態を見てみましょう。

```bash
# 現在のブランチの履歴を確認する
git log --oneline -3
# 出力:
# aaa1111 feat: データ拡張モジュールを追加（左右反転、回転、色の揺らぎ）
# bbb2222 README を改善：プロジェクト説明と使い方を追加
# ccc3333 .gitignore を追加

# main に切り替えて確認する
git checkout main

# main には augmentation.py がない！
ls src/
# model.py  train.py  utils.py  （augmentation.py はない）

# feature ブランチに戻る
git checkout feature/data-augmentation
ls src/
# augmentation.py  model.py  train.py  utils.py  （ある！）
```

これがブランチの便利さです。2 本の時間線が互いに影響しません。

---

## ブランチのマージ

ブランチ上の機能開発が終わってテストも通ったら、それを `main` にマージできます。

```bash
# 手順1: main ブランチに切り替える
git checkout main

# 手順2: feature ブランチを main にマージする
git merge feature/data-augmentation
```

出力：

```
Updating bbb2222..aaa1111
Fast-forward
 src/augmentation.py | 25 +++++++++++++++++++++++++
 src/train.py        |  5 +++++
 2 files changed, 30 insertions(+)
 create mode 100644 src/augmentation.py
```

これで `main` ブランチにもデータ拡張のコードが入りました。

```bash
ls src/
# augmentation.py  model.py  train.py  utils.py  ✅
```

### マージ後の整理

```bash
# 機能ブランチはもうマージされたので、削除してよい（リポジトリをきれいに保つ）
git branch -d feature/data-augmentation

# ブランチを確認する — main だけ残っている
git branch
# * main
```

---

## マージコンフリクト

### いつ起こる？

2 つのブランチが**同じファイルの同じ場所**を変更すると、Git はどちらを残すべきか判断できず、コンフリクトが起こります。

### 例：コンフリクトを起こして解決する

```bash
# main から 2 つのブランチを作り、2 人が同時に作業している状況を再現する
git checkout -b alice/update-model
cat > src/model.py << 'EOF'
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Alice: フィルター数を 32 に変更
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 16 * 16)
        return self.fc1(x)
EOF
git add . && git commit -m "alice: フィルター数を 32 に増やす"

# main に戻って、bob のブランチを作成する
git checkout main
git checkout -b bob/update-model
cat > src/model.py << 'EOF'
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # Bob: フィルター数を 64 にし、5x5 カーネルに変更
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 16 * 16)
        return self.fc1(x)
EOF
git add . && git commit -m "bob: 64 フィルターと 5x5 カーネルを採用"
```

まず Alice の変更をマージします。

```bash
git checkout main
git merge alice/update-model    # ✅ 成功、コンフリクトなし
```

次に Bob の変更をマージします。

```bash
git merge bob/update-model
# 出力:
# CONFLICT (content): Merge conflict in src/model.py
# Automatic merge failed; fix conflicts and then commit the result.
```

**コンフリクトが発生しました。** Alice と Bob が `model.py` の同じ行を変更していたからです。

### コンフリクトを解決する

`src/model.py` を開くと、Git がコンフリクト箇所を示しています。

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Alice: フィルター数を 32 に変更
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)
=======
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # Bob: フィルター数を 64 にし、5x5 カーネル
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)
>>>>>>> bob/update-model
```

- `<<<<<<< HEAD` から `=======` までが**現在のブランチ**（main、Alice の変更を含む）の内容
- `=======` から `>>>>>>> bob/update-model` までが**取り込もうとしているブランチ**（Bob）の内容

**最終的に何を残すかを手動で決める必要があります。** たとえば、ここでは Bob の案を採用するとします。

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, padding=2)  # Bob の案を採用
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 64 * 16 * 16)
        return self.fc1(x)
```

`<<<<<<<`、`=======`、`>>>>>>>` の印はすべて削除し、残したいコードだけにします。それから：

```bash
git add src/model.py
git commit -m "merge: Alice と Bob の変更を統合し、Bob の 64 フィルター案を採用"
```

これでコンフリクトは解決です。

:::tip VS Code でのコンフリクト解決
VS Code でコンフリクトが起きると、該当箇所が強調表示され、いくつかのボタンが出ます。
- **Accept Current Change**（現在のブランチの内容を残す）
- **Accept Incoming Change**（取り込もうとしている変更を残す）
- **Accept Both Changes**（両方残す）

クリックするだけで済むので、手動で直すよりずっと楽です。
:::

```bash
# ブランチを整理する
git branch -d alice/update-model
git branch -d bob/update-model
```

---

## Pull Request（知っておけば十分）

チーム開発では、通常 `main` ブランチへ直接マージしません。代わりに **Pull Request（PR）** を使い、まず他の人にコードを確認してもらい、問題がなければマージします。

### Pull Request の流れ

```
1. feature ブランチを作ってコードを書く
2. GitHub に push する
3. GitHub 上で Pull Request を作成する
4. 同僚がコードをレビューし、修正点を伝える
5. あなたが修正して、新しい commit を push する
6. 同僚が "Approve"（承認）する
7. コードが main ブランチにマージされる
```

### 実際の操作

```bash
# 1. ブランチを作ってコードを書く
git checkout -b feature/add-evaluation
echo "def evaluate(model, dataloader): pass" > src/evaluate.py
git add . && git commit -m "モデル評価モジュールを追加"

# 2. ブランチを GitHub に push する
git push -u origin feature/add-evaluation
```

その後 GitHub を開くと、次のような表示が出ます。

> feature/add-evaluation had recent pushes — **Compare & pull request**

このボタンをクリックして、PR のタイトルと説明を書き、**Create pull request** を押せば完了です。

個人プロジェクトなら、自分で確認したあと GitHub のページで **Merge pull request** を押してマージしてもかまいません。

---

## 本章のセルフチェック

以下を確認して、Git の基礎を身につけたか確かめましょう。

- [ ] ゼロから Git リポジトリを作成できる
- [ ] `add` → `commit` でコードを保存できる
- [ ] `git diff` で何が変更されたか確認できる
- [ ] `.gitignore` ファイルを書ける
- [ ] コードを GitHub に push できる
- [ ] `git clone` で他人のプロジェクトを取得できる
- [ ] ブランチの概念を理解し、作成とマージができる
- [ ] マージコンフリクトが起きても慌てず、解決方法が分かる

:::tip すべてチェックできましたか？
おめでとうございます。これで Git の学習は完了です。これらのスキルは、これからの AI 学習全体を通してずっと役立ちます。次は Python の開発環境を設定しましょう。
:::
