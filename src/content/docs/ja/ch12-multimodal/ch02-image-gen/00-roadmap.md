---
title: "12.2.1 画像生成ロードマップ：プロンプト、制御、レビュー"
description: "画像生成章を短く実践的に進めるための地図です。プロンプト記録、パラメータ保存、生成モード選択、出力レビューを先に練習します。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "画像生成ガイド, 拡散モデル, Stable Diffusion, ControlNet, LoRA"
---

# 12.2.1 画像生成ロードマップ：プロンプト、制御、レビュー

画像生成は、1 行のプロンプトで終わる作業ではありません。意図、プロンプト記録、パラメータ、必要な制御、候補比較、レビューまで含むワークフローです。

## まずパイプラインを見る

![画像生成章の学習フロー](/img/course/ch12-image-gen-chapter-flow-ja.webp)

![Stable Diffusion の応用モード選択](/img/course/ch12-sd-application-mode-selector-map-ja.webp)

![Stable Diffusion の微調整ルート選択](/img/course/ch12-sd-finetuning-route-choice-map-ja.webp)

最初の習慣は、何を作りたいか、どのモードを使ったか、どの seed やパラメータが結果を左右したか、出力前に何を確認すべきかを記録することです。

## プロンプト記録を作る

```python
import json

brief = {
    "topic": "RAG basics",
    "audience": "beginners",
    "style": "clean editorial cover",
}
prompt = f"{brief['style']} for {brief['topic']}, friendly visual metaphor for {brief['audience']}, clear layout"
record = {
    "mode": "text-to-image",
    "prompt": prompt,
    "negative_prompt": "blurry, watermark, unreadable text",
    "seed": 42,
    "review": ["legibility", "copyright", "brand safety"],
}

print(json.dumps(record, indent=2))
```

期待される出力：

```text
{
  "mode": "text-to-image",
  "prompt": "clean editorial cover for RAG basics, friendly visual metaphor for beginners, clear layout",
  "negative_prompt": "blurry, watermark, unreadable text",
  "seed": 42,
  "review": [
    "legibility",
    "copyright",
    "brand safety"
  ]
}
```

![画像生成 Prompt 記録の実行結果図](/img/course/ch12-image-prompt-record-result-map-ja.webp)

プロンプト記録を再現できなければ、画像を安定して改善することも難しくなります。

## この順番で学ぶ

| ステップ | 読む内容 | 練習の成果 |
|---|---|---|
| 1 | 拡散の直感 | ノイズ、デノイズ、seed、サンプリングを説明する |
| 2 | Stable Diffusion の部品 | text encoder、U-Net、VAE、latent space を図にする |
| 3 | 応用と制御 | text-to-image、image-to-image、inpainting、ControlNet、LoRA を比較する |

## 通過条件

プロンプト記録を書き、選んだ生成モードを説明し、3 つの候補メモを残し、出力前に少なくとも 1 つのレビューリスクを記録できれば、この章は通過です。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
プロンプト記録：プロンプト、否定条件、参照、seed/model、バージョン番号
候補出力：生成結果またはシミュレーション結果と選択理由
技術メモ：diffusion step、latent、cross-attention、LoRA、またはアプリケーションモード
失敗確認: プロンプトのずれ、文体不一致、成果物、著作権、肖像、またはレビュー失敗
期待される成果: 選定した画像/版の記録と却下候補のメモ
```
