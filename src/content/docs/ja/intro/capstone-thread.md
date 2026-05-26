---
title: "0.5 キャップストーンプロジェクト軸：コース知識アシスタント"
description: "1つのコース知識アシスタントで、第1-13章をポートフォリオ向け AI プロジェクトにつなげます。"
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIポートフォリオプロジェクト, AIフルスタックプロジェクト, RAGプロジェクト, Agentプロジェクト, オープンソースLLM配置"
---
![プロジェクト視点マップ](/img/course/appendix-ai-project-lens-map-ja.webp)

自分のプロジェクトがまだない場合は、**コース知識アシスタント**を標準プロジェクトにしてください。これは追加課題ではありません。各章で1つずつ能力を足し、最終的に説明でき、再実行でき、評価でき、配置できる AI アプリケーションへ育てるポートフォリオ軸です。

## 最終形

最終的に、このプロジェクトは次を満たします。

- コースノート、PDF、Web 抜粋、または自分の学習記録を読み込める。
- source、time、fields、quality notes を残しながらデータを整備できる。
- Prompt、RAG、Agent ワークフローで回答し、検索 trace と tool trace を保存できる。
- 固定評価質問、失敗サンプル、コスト/遅延メモ、安全境界を持つ。
- 必要に応じて画像、OCR、マルチモーダル素材、ローカルのオープンソースモデル実行環境を接続できる。
- レビューする人が README から中核経路を再実行できる。

## ディレクトリテンプレート

```tree
capstone-course-assistant/
  README.md
  data/
    raw/
    processed/
  notebooks/
  src/
    cli.py
    data_pipeline.py
    evals.py
    rag.py
    agent_tools.py
  reports/
    evidence_log.md
    failure_cases.md
    eval_results.csv
    runtime_notes.md
```

初日はフォルダと README だけで十分です。他のファイルは、章が進むにつれて自然に増えます。

## ポートフォリオ提出テンプレート

各大きな stage の終わりに、同じ package format を使います。これにより、project は demo の寄せ集めではなく、review できる成果物になります。

```text
README.md                  何をするか、どう実行するか、何をまだ支援しないか
run.sh or commands.md       正確な rerun path
data_note.md                source、fields、cleaning rules、privacy notes
eval_cases.csv              比較に使う fixed questions または inputs
failure_cases.md            少なくとも1つの honest failure と suspected cause
screenshots/ or outputs/    visible result、chart、trace、API response
release_note.md             この章で何が変わり、次に何を test するか
```

最小版は README、1つの run command、1つの output、1つの failure note です。強いポートフォリオ版は fixed eval set、before/after comparison、cost または latency note、safety boundary、短い demo script を含みます。

## 章ごとの成長

**第1-3章：再現可能な作業台**
環境コマンド、Git コミット、Python CLI、サンプルデータ、整備ルール、グラフ、データ品質メモを残します。

**第4-6章：モデルの証拠**
小さな分類、回帰、または表現学習実験で、baseline、指標、失敗サンプル、学習診断を練習します。高スコアではなく、証拠でモデルを判断することが目的です。

**第7章：LLM の挙動制御**
5-10 個の質問を固定し、Prompt、構造化出力、token/context 制限、失敗サンプルを比較します。任意で mini GPT-2 を動かし、学習と生成の経路を理解します。

**第8章：RAG による根拠つき回答**
コース資料を chunk に分け、metadata を付け、証拠を検索し、引用つき回答を生成します。最終回答を見る前に top-k chunk を保存します。

**第9章：Agent ツールループ**
ファイル読み取り、フォルダ一覧、レポート生成など、少数の安全な tool だけを公開します。tool schema、trace、安全ブロック、rollback メモを残します。

**第10-12章：プロダクト別の拡張**
画像や OCR なら第10章、ラベル・抽出・要約なら第11章、PDF・画像・音声・動画・クリエイティブパッケージなら第12章を使います。

**第13章：オープンソースモデル実行環境**
小型モデルから始め、ローカル推論、評価、OpenAI 風 API を動かします。GPU があれば vLLM や SGLang を試します。モデルライセンス、環境レポート、初回実行、評価表、停止手順を残します。

## 各章で1つだけ変える

各章の終わりに、次の4つへ答えます。

- プロジェクトにどの能力が増えたか。
- どのコマンドで再実行できるか。
- どの証拠が動作を示すか。
- どの失敗サンプルが主張を控えめにしてくれるか。

答えられない場合は、機能追加の前に証拠を足してください。

## 残す証拠

このページの学習結果を、プロジェクト軸の証拠カードとして保存します。

```text
プロジェクト名: コース知識アシスタント、または自分の代替プロジェクト
章ごとの成長ルール: demo を積み上げるのではなく、各章で1つの能力を足す
再実行経路: README コマンド、script、notebook cell、または service endpoint
レビュー用パッケージ: data note、eval cases、trace、failure note、release note
期待される成果: setup から RAG、Agent、runtime evidence まで育つ1本のプロジェクト軸
```

## 最低合格基準

メインルートを終えた後、このプロジェクトには少なくとも次が必要です。

- 実行可能な README。
- 小さなデータセットまたはドキュメントセット。
- 固定評価質問。
- Prompt/RAG/Agent trace。
- 失敗ケースと改善計画。
- クラウド API を使う時と、オープンソースモデル実行環境を使う時の説明。

目的は最大のシステムを作ることではありません。AI エンジニアリングのループを本当に理解していると、他の人が信じられるシステムを作ることです。

## 確認

この planning page は、project thread に 1 つの rerun command、1 つの evaluation artifact、1 つの known failure case があれば合格です。
