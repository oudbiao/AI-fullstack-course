# ch06 Vertical Image Refinement Handoff

Updated: 2026-05-19, Asia/Taipei.

This file records the current state of the Chapter 6 vertical image refinement so another agent can continue without restarting the audit.

## User Requirements

- Course audience: learners who know some Python/frontend and want to transition into AI.
- Learning outcome: learners should be able to build AI applications independently and understand model principles.
- Content should be beginner-friendly but not shallow: keep concepts, principles, runnable code, output reading, and project evidence.
- Images should prefer vertical teaching graphics when useful.
- Vertical images must be recomposed vertically, never stretched from landscape. Reject squeezed text, warped circles, narrow tables, or distorted arrows.
- Every generated image is only a candidate. Inspect it before replacing any official asset or adding a Markdown reference.
- Reject images with wrong language, tiny/gibberish text, invented metric/output values, unrelated decoration, or marketing-poster style.
- If content is dense, simplify or split into multiple images. Do not solve density by shrinking or distorting text.
- API key concurrency rule: 3 keys means 3 worker lanes. Each key runs one active image task; when that key finishes, its worker immediately pulls the next queued task. Do not wait for all 3 to finish before starting the next queued image.

## Files Already Changed

- `AGENT_HANDOFF.md`
  - Added image QA and per-key concurrency rules.
- `scripts/generate_course_images.py`
  - Added universal anti-stretch prompt rules.
  - Added ch06 vertical refinement prompts.
  - Added/updated per-key worker-lane parallel generation logic.
  - Added stricter prompt for `ch06-study-guide-training-loop.png`.
- `scripts/generate_localized_course_images.py`
  - Fixed localized HTTP call to pass `api_keys=[...]`.
  - Added localized anti-stretch instructions.
  - Added per-key worker-lane parallel generation logic.
- Chapter 6 docs and localized docs already have structure/content improvements from earlier in this workstream: evidence sections, expected outputs/results, stronger bridges, and more image/code/output anchors.

## Generated Candidate Location

Generated candidates remain available as review artifacts:

- `tmp/ch06-vertical-images/*.png`

The accepted candidate batch has already been converted into matching official WebP assets:

- `static/img/course/*.webp`

Current candidate count:

- 30 PNG files
- All are `1024x1792`
- Contact sheets:
  - `tmp/ch06-vertical-images/qa-contact-sheet-zh.png`
  - `tmp/ch06-vertical-images/qa-contact-sheet-en.png`
  - `tmp/ch06-vertical-images/qa-contact-sheet-ja.png`

## Current Candidate Set

- `ch06-attention-qkv-library-analogy-map.png`
- `ch06-attention-qkv-library-analogy-map-en.png`
- `ch06-attention-qkv-library-analogy-map-ja.png`
- `ch06-backprop-error-responsibility-map.png`
- `ch06-backprop-error-responsibility-map-en.png`
- `ch06-backprop-error-responsibility-map-ja.png`
- `ch06-causal-mask-no-peeking-map.png`
- `ch06-causal-mask-no-peeking-map-en.png`
- `ch06-causal-mask-no-peeking-map-ja.png`
- `ch06-deep-learning-project-cycle.png`
- `ch06-deep-learning-project-cycle-en.png`
- `ch06-deep-learning-project-cycle-ja.png`
- `ch06-neuron-linear-activation-gate.png`
- `ch06-neuron-linear-activation-gate-en.png`
- `ch06-neuron-linear-activation-gate-ja.png`
- `ch06-nn-module-parameter-flow.png`
- `ch06-nn-module-parameter-flow-en.png`
- `ch06-nn-module-parameter-flow-ja.png`
- `ch06-projects-portfolio-loop.png`
- `ch06-projects-portfolio-loop-en.png`
- `ch06-projects-portfolio-loop-ja.png`
- `ch06-pytorch-chapter-flow.png`
- `ch06-pytorch-chapter-flow-en.png`
- `ch06-pytorch-chapter-flow-ja.png`
- `ch06-study-guide-training-loop.png`
- `ch06-study-guide-training-loop-en.png`
- `ch06-study-guide-training-loop-ja.png`
- `ch06-transformer-chapter-flow.png`
- `ch06-transformer-chapter-flow-en.png`
- `ch06-transformer-chapter-flow-ja.png`

## QA State

Confirmed no active generation process at the time this handoff was written.

### Live Progress Log

- 2026-05-19 Asia/Taipei: Resumed from this handoff and `AGENT_HANDOFF.md`; confirmed no need to restart generation.
- 2026-05-19 Asia/Taipei: Recounted candidates in `tmp/ch06-vertical-images/`: 30 candidate PNGs plus contact sheets.
- 2026-05-19 Asia/Taipei: Verified every candidate PNG is `1024x1792`.
- 2026-05-19 Asia/Taipei: Verified every candidate PNG has a same-name target WebP in `static/img/course/`.
- 2026-05-19 Asia/Taipei: Confirmed current Chapter 6 Markdown references remain WebP paths; do not change them to PNG.
- 2026-05-19 Asia/Taipei: Confirmed the current official WebP versions for these 30 targets are still landscape `1536x1024`, so the accepted replacements will be real vertical updates.
- 2026-05-19 Asia/Taipei: Opened `tmp/ch06-vertical-images/qa-contact-sheet.png` and confirmed the batch is broadly vertical/compositional rather than obviously stretched landscape.
- 2026-05-19 Asia/Taipei: Opened original `ch06-backprop-error-responsibility-map-en.png`; QA pass. It is readable, instructional, vertically recomposed, and does not invent run metrics.
- 2026-05-19 Asia/Taipei: Opened original `ch06-backprop-error-responsibility-map-ja.png`; QA pass. It uses natural Japanese with acceptable technical English labels, no squeezed text, and no invented metrics.
- 2026-05-19 Asia/Taipei: Opened originals `ch06-causal-mask-no-peeking-map-en.png` and `ch06-causal-mask-no-peeking-map-ja.png`; QA pass. Both clearly show lower-triangular visibility, future masking, and a test analogy. English uses a small acceptable simplification around "past/current" but the drawing includes the current token and remains pedagogically correct.
- 2026-05-19 Asia/Taipei: Opened originals `ch06-neuron-linear-activation-gate-en.png` and `ch06-neuron-linear-activation-gate-ja.png`; QA pass. Both show linear weighted score, nonlinear activation, and output-to-next-layer flow without stretched text or fabricated metrics.
- 2026-05-19 Asia/Taipei: Opened originals `ch06-nn-module-parameter-flow-en.png` and `ch06-nn-module-parameter-flow-ja.png`; QA pass. Both show `nn.Module` as a trainable model container, forward flow, parameter/optimizer update, and train/eval mode. Text is readable and not stretched.
- 2026-05-19 Asia/Taipei: Opened originals `ch06-pytorch-chapter-flow-en.png` and `ch06-pytorch-chapter-flow-ja.png`; QA pass. Both are readable vertical roadmaps. The English version frames DataLoader before Autograd as data-supply context, while the Japanese version follows the article order exactly; both remain acceptable teaching maps.
- 2026-05-19 Asia/Taipei: Opened originals `ch06-transformer-chapter-flow-en.png` and `ch06-transformer-chapter-flow-ja.png`; QA pass. Both use readable classroom-note layouts covering RNN bottleneck, Attention, Q/K/V, Transformer block, and LLM bridge with no fabricated numeric results.
- 2026-05-19 Asia/Taipei: User paused this run to hand off to another agent. Do not continue QA/conversion from this agent unless explicitly resumed.
- 2026-05-19 Asia/Taipei: User explicitly resumed this run.
- 2026-05-19 Asia/Taipei: Opened original `ch06-deep-learning-project-cycle.png`; QA pass. It is a true vertical project-review loop, readable, not squeezed/stretched, and uses generic metrics/checkpoints without fabricated numeric results.
- 2026-05-19 Asia/Taipei: Wrote final QA report at `reports/course-images/ch06-vertical-refine/qa-report.md`; all 30 candidates are accepted for WebP conversion.
- 2026-05-19 Asia/Taipei: Converted all 30 accepted PNGs into same-name WebP files under `static/img/course/`, overwriting the previous landscape versions. Verified all converted WebPs are `1024x1792`.
- 2026-05-19 Asia/Taipei: Required validation commands passed: `python3 validate_markdown_fences.py`, `python3 validate_internal_links.py`, `python3 validate_sidebars.py`, `python3 validate_course_structure.py`, `python3 scripts/validate_course_image_refs.py`, and `git diff --check`.
- 2026-05-19 Asia/Taipei: `npm run build` passed for en, zh-Hans, and ja. Build post-processing ran successfully: `strip_build_null_bytes` checked 1314 HTML files and `merge_localized_sitemaps` merged 1305 URLs.
- 2026-05-19 Asia/Taipei: Final status check: all 30 target WebPs still verify as `1024x1792`; related files are modified/untracked as expected, with no staging or commit performed.
- 2026-05-19 Asia/Taipei: User said continue after completion; reconciled stale handoff/report wording so another agent sees the current completed state. No staging or commit performed.
- 2026-05-19 Asia/Taipei: Ran a light ch06 content sweep after the completed image work. Found no TODO/FIXME/TBD/placeholder markers and no `.png`/`.jpg` Markdown image references under ch06 English, zh-Hans, or Japanese docs. Confirmed all 44 files per locale have course images, code fences, and evidence anchors; expected-output wording exists with locale-specific variants such as `Expected shape`, `期望输出`, and `期待される形`.

### Pause / Handoff State

- Do not stage or commit.
- All 30 accepted candidate PNGs have been converted to official same-name WebP files in `static/img/course/`.
- Required validation/build commands have passed during this resumed run.
- Original-image QA is complete for all 30 candidates.
- QA report is written at `reports/course-images/ch06-vertical-refine/qa-report.md`.
- No staging or commit has been performed.
- If another agent takes over, the remaining decision is only whether to review/stage/commit the modified WebPs and reports.
- A post-completion ch06 content sweep found no TODO/placeholder markers and no PNG/JPG Markdown image references.

Images directly opened and visually checked as acceptable candidates:

- `ch06-backprop-error-responsibility-map.png`
- `ch06-backprop-error-responsibility-map-en.png`
- `ch06-backprop-error-responsibility-map-ja.png`
- `ch06-attention-qkv-library-analogy-map.png`
- `ch06-causal-mask-no-peeking-map.png`
- `ch06-causal-mask-no-peeking-map-en.png`
- `ch06-causal-mask-no-peeking-map-ja.png`
- `ch06-neuron-linear-activation-gate.png`
- `ch06-neuron-linear-activation-gate-en.png`
- `ch06-neuron-linear-activation-gate-ja.png`
- `ch06-nn-module-parameter-flow.png`
- `ch06-nn-module-parameter-flow-en.png`
- `ch06-nn-module-parameter-flow-ja.png`
- `ch06-pytorch-chapter-flow.png`
- `ch06-pytorch-chapter-flow-en.png`
- `ch06-pytorch-chapter-flow-ja.png`
- `ch06-transformer-chapter-flow.png`
- `ch06-transformer-chapter-flow-en.png`
- `ch06-transformer-chapter-flow-ja.png`
- `ch06-projects-portfolio-loop.png`
- `ch06-study-guide-training-loop.png`
- `ch06-study-guide-training-loop-en.png`
- `ch06-study-guide-training-loop-ja.png`
- `ch06-attention-qkv-library-analogy-map-en.png`
- `ch06-attention-qkv-library-analogy-map-ja.png`
- `ch06-deep-learning-project-cycle.png`
- `ch06-deep-learning-project-cycle-en.png`
- `ch06-deep-learning-project-cycle-ja.png`
- `ch06-projects-portfolio-loop-en.png`
- `ch06-projects-portfolio-loop-ja.png`

Important QA history:

- Old `ch06-study-guide-training-loop.png` failed because numbering jumped from 1 to 12. It has been regenerated and now uses continuous 1-8 steps.
- Old `ch06-study-guide-training-loop-ja.png` failed because it displayed concrete `accuracy`/`loss` numbers not tied to real code output. It has been regenerated and no longer shows fake metric values.
- New `ch06-study-guide-training-loop.png` and `ch06-study-guide-training-loop-ja.png` are visually acceptable candidates and have been recorded in the final QA report.

Still recommended before official adoption:

- No remaining original-image QA items from the previous list.

## Current Next Steps

1. Review the modified WebP assets and report files if desired.
2. Decide whether to stage and commit this completed ch06 vertical-image batch.
3. Keep Markdown image references pointing to WebP files; do not change them to PNG.
4. Re-run validations if any later file changes are made.

## Transfer Prompt

Use this prompt if handing the task to another agent:

```text
请接手 /Users/carl/Documents/AI-fullstack-course 的 ch06 图片与内容精修后续收尾。

先阅读：
1. /Users/carl/Documents/AI-fullstack-course/AGENT_HANDOFF.md
2. /Users/carl/Documents/AI-fullstack-course/reports/course-images/ch06-vertical-refine/HANDOFF.md
3. /Users/carl/Documents/AI-fullstack-course/reports/course-images/ch06-vertical-refine/qa-report.md

当前状态：
- 30 张 ch06 竖图候选 PNG 已逐张/分组 QA，通过后已转换为同名 WebP 覆盖 static/img/course/。
- 所有 30 个正式 WebP 已验证为 1024x1792。
- Markdown 图片引用保持 WebP，没有改成 PNG。
- required validations 和 npm run build 已通过。
- 继续后又清理了 handoff/report 的旧措辞，并做了 ch06 轻量内容扫尾：无 TODO/FIXME/TBD/placeholder，ch06 三语 Markdown 无 PNG/JPG 图片引用。
- 目前未 stage、未 commit。不要提交 scripts/__pycache__ 或临时无关文件。

下一步只需要根据用户指示决定是否 review/stage/commit：
- review 范围重点是 static/img/course/ch06-*.webp、reports/course-images/ch06-vertical-refine/、必要时 tmp/ch06-vertical-images/ 作为审阅材料。
- 如果用户继续要求内容精修，先不要从头生成图片；从 handoff 的完成状态继续。
- 如果做任何新改动，更新 reports/course-images/ch06-vertical-refine/HANDOFF.md，并重新跑必要验证。
```

Validation commands already passed in this resumed run:

```bash
python3 validate_markdown_fences.py
python3 validate_internal_links.py
python3 validate_sidebars.py
python3 validate_course_structure.py
python3 scripts/validate_course_image_refs.py
git diff --check
npm run build
```

Optional ch06 audit command:

```bash
python3 - <<'PY'
from pathlib import Path

roots = {
    "en": Path("docs/ch06-deep-learning"),
    "zh": Path("i18n/zh-Hans/docusaurus-plugin-content-docs/current/ch06-deep-learning"),
    "ja": Path("i18n/ja/docusaurus-plugin-content-docs/current/ch06-deep-learning"),
}

for locale, root in roots.items():
    files = sorted(root.rglob("*.md"))
    evidence = image = code = expected = 0
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        evidence += int(any(marker in text for marker in ["Evidence to Keep", "保留证据", "残す証拠"]))
        image += int("](/img/course/" in text)
        code += int("```" in text)
        expected += int(any(marker in text for marker in ["Expected", "预期", "期待"]))
    print(locale, "files=", len(files), "evidence=", evidence, "image=", image, "code=", code, "expected=", expected)
PY
```

## Do Not Forget

- Do not stage or commit unless the user asks.
- Do not commit `scripts/__pycache__`.
- Temporary files under `tmp/` are review artifacts, not course assets.
- Keep English, Simplified Chinese, and Japanese image sets aligned.
- If any generated image feels “pretty but not instructional,” reject it.
