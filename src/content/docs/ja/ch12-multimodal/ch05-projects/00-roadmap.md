---
title: "12.5.1 総合プロジェクトロードマップ：クリエイティブパッケージのワークフロー"
description: "AIGC 総合プロジェクト章を短く実践的に進めるための地図です。brief をコピー、画像プロンプト、動画台本、素材バージョン、レビュー、出力へ変えます。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIGCプロジェクトガイド, クリエイティブプラットフォーム, マルチモーダルプロジェクト, コンテンツ生成ワークフロー"
---
卒業プロジェクトは、多くのモデル API をつなぐことが目的ではありません。ユーザーが brief を入力し、素材を生成し、バージョンを比較し、編集し、レビューし、使えるコンテンツパッケージとして出力できる流れを作ることです。

## まずプロダクトの閉ループを見る

![AIGC クリエイティブプラットフォームの納品ループ](/img/course/ch12-projects-delivery-loop-ja.webp)

![クリエイティブパッケージのパイプライン](/img/course/ch12-workshop-creative-package-pipeline-map-ja.webp)

![プロンプト、素材、バージョンの地図](/img/course/ch12-workshop-prompt-asset-version-map-ja.webp)

![レビューと出力の地図](/img/course/ch12-workshop-review-export-map-ja.webp)

最初の習慣は、生成結果をすべて素材として保存し、出所、プロンプト、バージョン、レビュー状態、出力先を付けることです。

## 最小パッケージ状態を作る

```python
brief = {
    "topic": "RAG mini course",
    "audience": "new learners",
}
package = {
    "brief_ready": True,
    "assets": ["title", "cover_prompt", "video_script", "review_checklist"],
    "has_versions": True,
    "has_review": True,
}

ready = package["brief_ready"] and package["has_versions"] and package["has_review"] and len(package["assets"]) >= 4

print("package_ready:", ready)
print("assets:", ", ".join(package["assets"]))
```

期待される出力：

```text
package_ready: True
assets: title, cover_prompt, video_script, review_checklist
```

![最小パッケージ状態の実行結果図](/img/course/ch12-package-state-readiness-result-map-ja.webp)

この状態構造がないと、プロジェクトはプロダクトではなくデモに見えやすくなります。

## まずワークショップから始める

大きなクリエイティブプラットフォームへ広げる前に、[12.5.3 実践：再現可能なマルチモーダルクリエイティブパッケージを作る](./02-hands-on-multimodal-workshop.md) を先に動かしましょう。brief 受付、プロンプト記録、素材バージョン、絵コンテ出力、安全レビュー、失敗分析の最小閉ループができます。

## プロジェクト納品基準

| 納品物 | 最低要件 |
|---|---|
| README | 目的、実行コマンド、依存関係、素材の出所、例を書く |
| コンテンツパッケージ例 | 1 つの完全な brief、生成素材、レビュー記録を含める |
| バージョン記録 | 少なくとも 2 つの候補出力、または 1 回の編集履歴を残す |
| 安全レビュー | 著作権、肖像、声、センシティブ内容、出力ラベルを見る |
| 失敗記録 | 実際の失敗例 1 つと次の修正計画を書く |

## 通過条件

プロジェクトが brief を受け取り、構造化されたクリエイティブパッケージを作り、バージョンを残し、レビューを通し、他の人が確認できる Markdown または JSON を出力できれば、この章は通過です。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
要約：ユーザーの目的、対象読者、素材、制約、出力形式
成果物: ソースファイル、プロンプト、生成候補、選択出力、却下版
レビュー: 事実確認、著作権・肖像権・機微情報チェック、人の判断
統合：RAG レコード、Agent トレース、クリエイティブパッケージ、ストーリーボード、またはエクスポートプレビュー
期待される成果: README、レビュー用チェックリスト、失敗メモを含む再現可能なアセットパッケージ
```
