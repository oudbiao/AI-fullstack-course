# Chapter 5 Kaggle Validation Image QA

Date: 2026-05-24

## Scope

Regenerated the trilingual `ch05-kaggle-validation-leaderboard-loop` image set because the prior assets read like pasted UI cards on an office background instead of the project-required hand-drawn teaching style.

Accepted candidate PNGs:

- `tmp/ch05-kaggle-validation-whiteboard/ch05-kaggle-validation-leaderboard-loop.png`
- `tmp/ch05-kaggle-validation-whiteboard/ch05-kaggle-validation-leaderboard-loop-en.png`
- `tmp/ch05-kaggle-validation-whiteboard/ch05-kaggle-validation-leaderboard-loop-ja.png`

Published WebP assets:

- `public/img/course/ch05-kaggle-validation-leaderboard-loop.webp`
- `public/img/course/ch05-kaggle-validation-leaderboard-loop-en.webp`
- `public/img/course/ch05-kaggle-validation-leaderboard-loop-ja.webp`

## QA Result

All three accepted candidates are native vertical teaching images at `1024x1792`.

Visual checks passed:

- Hand-drawn whiteboard / lined-notebook teaching style, not SVG-style infographic layout.
- No office-photo background, pasted UI card stack, real Kaggle logo, website screenshot, fake score, rank, watermark, or tiny filler text.
- The workflow teaches Local CV -> one change -> `submission.csv` -> Public LB -> diagnose gap -> experiment log.
- The Japanese subtitle uses the corrected wording `提出はスコアだけでなく、実験記録でもある。`.
- English, Simplified Chinese, and Japanese versions keep the same teaching purpose and visual rhythm.

Rejected or regenerated candidates:

- The first English API attempt failed with upstream `502`; no partial output was used.

## Reference Check

After the change, the three Kaggle project pages still reference the same localized WebP filenames. Only the image contents and reproducible generation prompt were updated.
