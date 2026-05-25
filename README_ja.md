# AI Roads

<div align="center">
  <img src="./public/img/logo.svg" width="96" alt="AI Roads logo">
  <h3>実践者のための無料多言語 AI エンジニアリングカリキュラム。</h3>
  <p>
    <a href="./README.md">English</a> |
    <a href="./README_zh.md">简体中文</a> |
    日本語
  </p>
  <p>
    <a href="https://airoads.org">公式サイト</a>
  </p>
</div>

---

AI Roads は、実践重視の無料 AI エンジニアリングカリキュラムです。開発者基礎から始まり、Python、データ分析、AI 数学、機械学習、深層学習、LLM 原理、RAG、AI Agent、マルチモーダル AIGC、オープンソース LLM のデプロイと微調整へ進みます。

このリポジトリは公開学習サイト [airoads.org](https://airoads.org) を構成しています。サイトの既定言語は英語で、簡体字中国語と日本語のルートも用意しています。

## プロジェクトの目的

AI Roads はリンク集ではなく、プロジェクトを作りながら進む学習ルートです。重視していることは次の通りです。

1. ツール基礎から AI プロジェクトまでの明確な順序。
2. コマンド、コード、出力、残す証拠を含む実行可能なレッスン。
3. 図解、漫画、結果マップ、ローカライズ画像による理解支援。
4. 学習内容をポートフォリオに変えるプロジェクトチェックポイント。
5. 英語、簡体字中国語、日本語の同期。
6. 検証、ビルド、SEO、デプロイを支えるスクリプト。

読むだけで終わらせず、各段階で「動くもの」「説明できるもの」「見せられるもの」を残すことを目標にします。

## 対象読者

- AI エンジニアリングに入りたい初学者。
- プログラミング経験があり、AI を体系的に学びたい開発者。
- 概念だけでなくプロジェクトを作りたい学習者。
- LLM、RAG、Agent、CV、NLP、マルチモーダルのポートフォリオを作りたい学生。
- 多言語の静的ドキュメントサイトを運用したいメンテナー。

## コースロードマップ

```text
0   スタートガイド
1   開発者ツール
2   Python プログラミング
3   データ分析と可視化
4   AI のための最小数学基礎
5   機械学習
6   深層学習と Transformer 基礎
7   LLM 原理、Prompt、微調整
8   LLM アプリ開発と RAG
9   AI Agent と Agentic Systems
10  コンピュータビジョン
11  LLM 後の NLP 専門
12  AIGC とマルチモーダル
13  オープンソース LLM のデプロイと微調整
E   選択モジュール
A   付録
```

まず 1-9 章を進め、その後プロジェクト方向に応じて 10-13 章を選ぶのがおすすめです。

## 主な学習ステーション

- **1 開発者ツール**：ターミナル、Git、ローカル開発環境。
- **2 Python**：文法、データ構造、ファイル、OOP、API、プロジェクト。
- **3 データ分析**：NumPy、Pandas、可視化、データベース。
- **4 AI 数学**：線形代数、確率、微積分、最適化。
- **5 機械学習**：教師あり学習、教師なし学習、評価、特徴量設計。
- **6 深層学習**：PyTorch、ニューラルネット、CNN、RNN、Transformer、生成モデル。
- **7 LLM 原理**：NLP 基礎、Transformer、事前学習、Prompt、微調整、アラインメント。
- **8 RAG 応用**：文書処理、ベクトル DB、検索、評価、デプロイ。
- **9 AI Agent**：計画、ツール、メモリ、MCP、マルチ Agent、可観測性、安全性。
- **10 Computer Vision**：分類、検出、セグメンテーション、OCR、動画、3D vision。
- **11 NLP 専門**：テキスト表現、分類、抽出、Seq2Seq、事前学習モデル。
- **12 マルチモーダル AIGC**：視覚言語モデル、画像/動画/音声生成、倫理、製品プロトタイプ。
- **13 オープンソース LLM**：ローカル CPU、無料 Colab、レンタル GPU の3ルート、モデル実行、サービス化、評価、GPU 利用計画、LoRA 判断。

## 技術スタック

| レイヤー | 選択 |
|---|---|
| 静的サイトフレームワーク | Astro 6 |
| ドキュメント UI | Astro Starlight |
| コンテンツ形式 | Markdown / MDX-compatible Starlight docs |
| 検索 | Starlight Pagefind integration |
| 多言語 | 英語ルート、簡体字中国語 `/zh-cn/`、日本語 `/ja/` |
| 本番ドメイン | `https://airoads.org` |
| コース素材 | `public/img/course/` |
| 検証 | Markdown、内部リンク、sidebar、コース構造、図表、生成サイト QA |

## ローカル開発

Node.js 18 以上を使います。

```bash
npm install
npm run dev
```

静的サイトをビルドします。

```bash
npm run build
```

ビルド結果をプレビューします。

```bash
npm run serve
```

## 品質チェック

よく使うコマンド：

```bash
npm run build
npm run qa:diagrams
npm run qa:dist
npm run qa:course
npm run seo:indexnow:dry-run
```

`npm run qa:course` は、対応しやすいコース本文の不足を報告します。appendix、navigation page、study guide は folded-answer advisory から除外し、残るサンプルが walkthrough を補うべき lesson page を指すようにしています。

直接検証する場合：

```bash
python3 validate_markdown_fences.py
python3 validate_internal_links.py
python3 validate_sidebars.py
python3 validate_course_structure.py
python3 scripts/validate_course_image_refs.py
```

## ディレクトリ構成

```text
src/content/docs/        英語、簡体字中国語、日本語のコース本文
public/img/course/       コース図解、漫画、ローカライズ画像
public/img/logo.svg      AI Roads の公開ロゴ
public/img/social-card.png
astro.config.mjs         Astro Starlight 設定、言語、sidebar、sitemap、metadata
scripts/                 検証、sitemap、画像生成、SEO、保守スクリプト
docker/                  Docker デプロイ用 Nginx 設定
nginx/                   本番 proxy の例
```

## ビジュアル素材

コース画像は学習内容の一部であり、`public/img/course/` に置かれています。図解、結果マップ、漫画、ローカライズ画像が含まれます。画像を追加または置き換える場合は、近くのレッスン、コード、出力証拠と密接に結びつけてください。

## コントリビューション

Issue と Pull Request を歓迎します。特に歓迎する改善は次の通りです。

- 分かりにくいレッスンや壊れたリンクの修正。
- 実行可能な例、期待出力、トラブルシューティングの追加。
- 英語、簡体字中国語、日本語の同期改善。
- コース図解やローカライズ画像の改善。
- 検証、デプロイ、SEO スクリプトの強化。

コース本文を変更する場合は、3つの言語ルートで構造と意味をそろえてください。

## ライセンス

コース本文とプロジェクトコードは MIT License で公開されています。
