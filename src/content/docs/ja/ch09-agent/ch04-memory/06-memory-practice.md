---
title: "9.4.7 実践：完全な記憶システム"
description: "実行できる多層記憶 Agent を作ります。短期ウィンドウ、長期の好み、エピソード記録、手続き記憶が連携し、問い合わせ、書き込み、圧縮、返信生成までの一連の流れを示します。"
sidebar:
  order: 24
head:
  - tag: meta
    attrs:
      name: keywords
      content: "memory practice, short term, long term, episodic memory, procedural memory, agent"
---
:::tip[この節の位置づけ]
前のいくつかの節では、記憶を概念と戦略に分けて学びました。  
この節では、すぐ動かせる「小さなシステム」を直接作って、これらの層をつなげます。

- 短期記憶：最近の会話と現在の状態
- 長期記憶：ユーザーの好みと安定した情報
- エピソード記憶：タスクを処理したときの履歴
- 手続き記憶：固定のワークフロー手順

目標は大規模で全部入りのものを作ることではなく、まず「完全な記憶の閉ループ」を動かせるようにすることです。
:::
## 学習目標

- 複数層の記憶を同じ Agent の状態機械に入れる方法を学ぶ
- 「いつ、どの層の記憶に書き込むか」のルールを設計する方法を学ぶ
- 記憶を保存するだけでなく、実際の回答に参加させる方法を学ぶ
- 実行可能なサンプルを通して、再利用しやすいプロジェクトの土台を作る

---

## 作るシステムはどんなもの？

### 対象シナリオ

ここでもサポート対応のアシスタント場面を使います。  
ユーザーは続けて次のようなことを聞きます。

- 返金条件
- 返金の進捗
- 回答スタイルの指定

システムには、次の2つを実現してほしいです。

1. 現在の会話の中で一貫性を保つ
2. 次に来たときもユーザーの好みを覚えている

### 4層の記憶の役割分担

この例では、次のように分けます。

- `short_term`  
  最近の N 件のメッセージ + 現在のタスク状態
- `long_term`  
  ユーザーの長期的な好み
- `episodic`  
  毎回タスクを処理した後の要約エントリ
- `procedural`  
  あらかじめ定義したフローテンプレート。たとえば返金処理の手順

### 評価の目標

この実践サンプルで最も大事な確認ポイントは次の通りです。

- 好みを正しく書き込めるか
- 後の回答でその好みを参照できるか
- 検索できる情景記録を残せるか
- 回答前に手続き記憶の流れを参照できるか

---

## まずは完全に実行できる版を動かす

次のコードは 2 回の会話をシミュレーションします。

1. 1 回目でユーザーが「簡潔に答えて」と言い、返金条件を尋ねる
2. 2 回目でユーザーが進捗を尋ねると、システムは自動的に簡潔なスタイルを引き継ぐ

出力されるもの：

- 返信結果
- 4 層の記憶スナップショット

```python
from collections import deque
from dataclasses import dataclass, asdict


def get_refund_policy():
    return "返金ルール：購入後7日以内かつ学習進度が20%未満なら返金申請が可能です。代金は元の支払い方法に返金され、通常3〜7営業日で着金します。"


def get_order_status(order_id):
    mock = {
        "ORD-1001": {"status": "未発送", "progress": 0.12, "amount": 299},
        "ORD-1002": {"status": "発送済み", "progress": 0.35, "amount": 499},
    }
    return mock.get(order_id, {"status": "不明", "progress": None, "amount": None})


@dataclass
class Episode:
    user_id: str
    topic: str
    summary: str


class MemoryAgent:
    def __init__(self, short_window=4):
        self.short_term_messages = deque(maxlen=short_window)
        self.short_term_state = {}
        self.long_term_profile = {}
        self.episodic_memory = []
        self.procedural_memory = {
            "refund_workflow": [
                "注文状態を読み取る",
                "返金ポリシーを読み取る",
                "条件を満たすか判断する",
                "結論と着金の説明を返す",
            ]
        }

    def _remember_short(self, role, content):
        self.short_term_messages.append({"role": role, "content": content})

    def _update_profile(self, user_id, message):
        if "簡潔" in message:
            self.long_term_profile.setdefault(user_id, {})["style"] = "concise"
        if "詳細" in message:
            self.long_term_profile.setdefault(user_id, {})["style"] = "detailed"

    def _style_for_user(self, user_id):
        return self.long_term_profile.get(user_id, {}).get("style", "default")

    def _format_answer(self, text, style):
        if style == "concise":
            return text[:70] + ("..." if len(text) > 70 else "")
        if style == "detailed":
            return text + " よろしければ、具体的な操作手順やよくある失敗原因も補足できます。"
        return text

    def _write_episode(self, user_id, topic, summary):
        self.episodic_memory.append(Episode(user_id=user_id, topic=topic, summary=summary))

    def handle(self, user_id, user_message, order_id):
        self._remember_short("user", user_message)
        self._update_profile(user_id, user_message)

        self.short_term_state["active_workflow"] = "refund_workflow"
        self.short_term_state["order_id"] = order_id

        workflow = self.procedural_memory["refund_workflow"]
        order_info = get_order_status(order_id)
        policy = get_refund_policy()

        if order_info["status"] == "不明":
            answer = "その注文はまだ確認できません。注文番号を確認して、もう一度お試しください。"
        elif order_info["progress"] is not None and order_info["progress"] < 0.2:
            answer = (
                f"注文 {order_id} の学習進度は現在 {order_info['progress']*100:.0f}% です。"
                f"返金条件を満たしています。{policy}"
            )
        else:
            answer = (
                f"注文 {order_id} の学習進度は現在 {order_info['progress']*100:.0f}% です。"
                "すでに返金の進度しきい値を超えているため、今は直接返金の条件を満たしていません。"
            )

        style = self._style_for_user(user_id)
        final_answer = self._format_answer(answer, style)
        self._remember_short("assistant", final_answer)

        self._write_episode(
            user_id=user_id,
            topic="refund",
            summary=f"workflow={workflow}; order={order_id}; style={style}; result={final_answer}",
        )

        return final_answer

    def snapshot(self, user_id):
        return {
            "short_term_messages": list(self.short_term_messages),
            "short_term_state": dict(self.short_term_state),
            "long_term_profile": self.long_term_profile.get(user_id, {}),
            "episodic_memory_tail": [asdict(x) for x in self.episodic_memory[-2:]],
            "procedural_memory": self.procedural_memory,
        }


agent = MemoryAgent(short_window=4)

print("round1:")
print(agent.handle("u_001", "簡潔に答えてください。返金条件を知りたいです", "ORD-1001"))
print("\nround2:")
print(agent.handle("u_001", "では、いつ着金しますか？", "ORD-1001"))

print("\nmemory snapshot:")
print(agent.snapshot("u_001"))
```

想定出力：

```text
round1:
注文 ORD-1001 の学習進度は現在 12% です。返金条件を満たしています。返金ルール：購入後7日以内かつ学習進度が20%未満なら返金...

round2:
注文 ORD-1001 の学習進度は現在 12% です。返金条件を満たしています。返金ルール：購入後7日以内かつ学習進度が20%未満なら返金...

memory snapshot:
{'short_term_messages': [{'role': 'user', 'content': '簡潔に答えてください。返金条件を知りたいです'}, {'role': 'assistant', 'content': '注文 ORD-1001 の学習進度は現在 12% です。返金条件を満たしています。返金ルール：購入後7日以内かつ学習進度が20%未満なら返金...'}, {'role': 'user', 'content': 'では、いつ着金しますか？'}, {'role': 'assistant', 'content': '注文 ORD-1001 の学習進度は現在 12% です。返金条件を満たしています。返金ルール：購入後7日以内かつ学習進度が20%未満なら返金...'}], 'short_term_state': {'active_workflow': 'refund_workflow', 'order_id': 'ORD-1001'}, 'long_term_profile': {'style': 'concise'}, 'episodic_memory_tail': [{'user_id': 'u_001', 'topic': 'refund', 'summary': "workflow=['注文状態を読み取る', '返金ポリシーを読み取る', '条件を満たすか判断する', '結論と着金の説明を返す']; order=ORD-1001; style=concise; result=注文 ORD-1001 の学習進度は現在 12% です。返金条件を満たしています。返金ルール：購入後7日以内かつ学習進度が20%未満なら返金..."}, {'user_id': 'u_001', 'topic': 'refund', 'summary': "workflow=['注文状態を読み取る', '返金ポリシーを読み取る', '条件を満たすか判断する', '結論と着金の説明を返す']; order=ORD-1001; style=concise; result=注文 ORD-1001 の学習進度は現在 12% です。返金条件を満たしています。返金ルール：購入後7日以内かつ学習進度が20%未満なら返金..."}], 'procedural_memory': {'refund_workflow': ['注文状態を読み取る', '返金ポリシーを読み取る', '条件を満たすか判断する', '結論と着金の説明を返す']}}
```

![MemoryAgent 4層メモリ snapshot 結果図](/img/course/ch09-memory-four-layer-snapshot-result-map-ja.webp)

### このコードは、4 層がどう協力しているかをどう表している？

1. `short_term_messages`  
   直近の会話を保持する
2. `long_term_profile`  
   ユーザーの話し方の好みを覚える
3. `episodic_memory`  
   毎回タスクが終わるたびに「経験記録」を 1 件残す
4. `procedural_memory`  
   返金タスクの流れテンプレートを定義する

この 4 層すべてが使われていて、もう「概念を説明しただけで動いていない」状態ではありません。

### なぜ 2 回目でも簡潔なスタイルを保てるの？

1 回目でユーザーが「簡潔に答えてください」と言ったので、  
システムはそれを長期的な好みに書き込みます。

- `long_term_profile["u_001"]["style"] = "concise"`

そのため、2 回目にユーザーが同じことを繰り返さなくても、返信はそのスタイルを引き継ぎます。

### エピソード記憶はここでどんな価値があるの？

毎回の処理が終わるたびに、システムは episode summary を 1 件書きます。  
これにより、後から次のようなことに答えられます。

- ユーザーはこれまでにどんな返金判断を経験したか
- そのときの判断根拠は何だったか

これは振り返りや説明にとても役立ちます。

---

## このシステムはどう拡張できる？

### 長期記憶に「信頼度」と「更新時刻」を加える

とても古い、または信頼度の低い情報が、いつまでも回答に影響し続けるのを防げます。

### エピソード記憶に検索機能を加える

たとえば topic やキーワードで過去の経験を探せるようにすると、  
複雑な問題に対して過去の参照を付けられます。

### 手続き記憶をバージョン管理する

フローが変わったときに、次のような追跡ができます。

- どの会話でどの版のフローを使ったか

これは監査や再現にとても重要です。

---

## 実践で起こりやすい落とし穴

### すべての情報を長期記憶に書き込んでしまう

その結果、次のようになります。

- 検索ノイズがどんどん増える

### 「書き込みの基準」がない

たとえば、ユーザーの何気ない一言をそのまま長期保存すると、  
システムは間違った好みを学習しやすくなります。

### 記憶は保存するだけで、判断に使わない

この場合、システムは「記憶がある」ように見えますが、  
実際には回答が何も変わりません。

---

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
メモリ種別：短期、長期、エピソード記憶、または手続き記憶
書き込みルール：メモリが作成または更新されるとき
取得ルール：クエリ、関連性、鮮度、権限チェック
失敗確認: 古い記憶、プライバシー漏えい、矛盾、または過剰検索
クリーンアップ操作：要約、統合、期限切れ、削除、または確認を求める
```

## まとめ

この節で最も大事なのは、「完全な記憶システム」を実行可能な閉ループとして形にすることです。

> **短期は今のタスクを維持し、長期は安定した好みを残し、情景は過去の経験を蓄積し、手続きは再利用できるフローを覚える。**

この 4 層が連携して初めて、Agent は「一回きりの Q&A ツール」から「継続的に使えるタスクシステム」になります。

---

## 練習

1. サンプルに `user_blacklist_topic` という長期的な好みを追加し、システムが回答中に関係ない話題を避けられるか試してください。
2. `episodic_memory` が `topic` ごとに最近 1 件を検索できるようにしてください。
3. `procedural_memory` を複数フロー版に変えてください。たとえば `refund_workflow` と `invoice_workflow` です。
4. 考えてみましょう。どんな情報は短期だけに置くのが最適で、長期には置かないほうがよいでしょうか。

<details>
<summary>参考実装と解説</summary>

1. `user_blacklist_topic` は明示的な長期 preference として保存し、scope を明確にします。無関係な提案を抑えるために使い、安全上必要な情報やタスク情報まで遮断してはいけません。
2. `topic` で最新 episode を取るには、topic で filter し、timestamp または単調増加 id で sort します。
3. multi-workflow の procedural memory は、`refund_workflow` や `invoice_workflow` など workflow name を key にした辞書にできます。それぞれ steps と risk gates を持ちます。
4. 一回限りの制約、一時的な goal、現在の tool result、draft choice、セッション内だけに置くべき sensitive information は、長期記憶ではなく短期記憶に置きます。

</details>
