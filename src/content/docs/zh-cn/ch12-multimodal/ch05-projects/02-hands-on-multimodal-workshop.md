---
title: "12.5.3 实操：构建一个可复现的多模态创意包"
description: "AIGC 与多模态实操工作坊：本地构建创意包流水线，覆盖需求输入、Prompt 记录、SVG 视觉资产、分镜导出、资产审核、安全检查和失败分析。"
sidebar:
  order: 20
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIGC 实操, 多模态项目, Prompt 记录, 资产审核, 安全审核, 创意包"
---
在连接真实图像、视频或语音模型之前，先做一个小工作流，证明你理解产品闭环：输入、Prompt 规划、资产生成、版本记录、审核、导出和失败案例。

![多模态创意包流水线图](/img/course/ch12-workshop-creative-package-pipeline-map.webp)

这个练习只使用 Python 标准库，并在本地生成 SVG 模拟资产。这样做是故意的：目标是先让工作流可复现，再把 SVG baseline 替换成图像生成、TTS、视频生成或多模态模型 API。

## 你会构建什么

脚本会创建这个文件夹：

```text
multimodal_workshop_run/
  inputs/
    creative_brief.json
  prompts/
    prompt_plan.json
    prompt_versions.md
  assets/
    scene_01.svg
    scene_02.svg
    scene_03.svg
  outputs/
    storyboard.json
    timeline.csv
    content_package.json
    export_preview.html
  reports/
    asset_manifest.csv
    safety_review.md
    failure_cases.md
  README.md
```

最重要的结果不是 SVG 本身，而是每个生成资产都有 Prompt、来源记录、审核结果、导出边界和失败说明。

## 步骤 0：先看懂产品闭环

实操闭环是：

1. 写清创意 brief。
2. 把 brief 拆成分镜级 Prompt。
3. 生成 baseline 视觉资产。
4. 生成 storyboard 和 timeline。
5. 检查资产来源、授权、人像风险、对比度和导出限制。
6. 导出 HTML 预览和内容包。
7. 保存失败案例，供下一轮迭代。

## 步骤 1：创建文件夹和脚本

```bash
mkdir multimodal-workshop
cd multimodal-workshop
python3 -m venv .venv
source .venv/bin/activate
```

不需要 `pip install`。创建 `multimodal_workshop.py`，粘贴下面完整脚本。

![Prompt 到资产版本记录图](/img/course/ch12-workshop-prompt-asset-version-map.webp)

## 步骤 2：运行完整脚本

```python
from __future__ import annotations

import csv
import html
import json
import shutil
from pathlib import Path

RUN_DIR = Path("multimodal_workshop_run")
if RUN_DIR.exists():
    shutil.rmtree(RUN_DIR)
INPUT_DIR = RUN_DIR / "inputs"
PROMPT_DIR = RUN_DIR / "prompts"
ASSET_DIR = RUN_DIR / "assets"
OUTPUT_DIR = RUN_DIR / "outputs"
REPORT_DIR = RUN_DIR / "reports"
for folder in [INPUT_DIR, PROMPT_DIR, ASSET_DIR, OUTPUT_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

BRIEF = {
    "project_id": "course-launch-multimodal-kit",
    "topic": "AI learning assistant launch kit",
    "audience": "beginners learning AI full-stack development",
    "goal": "create a small content package for a course landing page and short video storyboard",
    "tone": "clear, practical, optimistic",
    "deliverables": ["poster SVG", "three-shot storyboard", "review checklist", "export preview HTML"],
    "constraints": ["no real person likeness", "no external copyrighted assets", "show source and review records"],
}

SCENES = [
    {
        "id": "scene_01",
        "title": "From scattered materials to one learning assistant",
        "visual_prompt": "A clean course dashboard combining text notes, screenshots, and voice notes into one AI learning assistant workspace.",
        "copy": "Turn scattered study materials into a guided AI learning workflow.",
        "duration_sec": 5,
        "background": "#f7f3e8",
        "accent": "#2563eb",
        "text_color": "#1f2937",
        "source": "script_generated_svg",
        "license": "generated_by_script_for_course_demo",
        "uses_real_person": False,
    },
    {
        "id": "scene_02",
        "title": "Review before export",
        "visual_prompt": "A multimodal review desk with prompt versions, asset records, copyright checks, and export options.",
        "copy": "Save prompts, assets, human review, and export limits before sharing.",
        "duration_sec": 6,
        "background": "#e8eef7",
        "accent": "#93c5fd",
        "text_color": "#bfdbfe",
        "source": "script_generated_svg",
        "license": "generated_by_script_for_course_demo",
        "uses_real_person": False,
    },
    {
        "id": "scene_03",
        "title": "Ready for a portfolio demo",
        "visual_prompt": "A final portfolio package showing poster, storyboard, safety checklist, and README ready for presentation.",
        "copy": "A usable AIGC project is a workflow, not a single pretty output.",
        "duration_sec": 5,
        "background": "#ecfdf5",
        "accent": "#059669",
        "text_color": "#064e3b",
        "source": "script_generated_svg",
        "license": "generated_by_script_for_course_demo",
        "uses_real_person": False,
    },
]

REVIEW_RULES = [
    "source_recorded",
    "license_recorded",
    "no_real_person_likeness",
    "sufficient_contrast",
    "export_limits_written",
]


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def relative_luminance(color: str) -> float:
    def channel(value: int) -> float:
        value = value / 255
        return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(v) for v in hex_to_rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(left: str, right: str) -> float:
    l1 = relative_luminance(left)
    l2 = relative_luminance(right)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def wrap_text(text: str, width: int = 46) -> list[str]:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if len(trial) > width and current:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def svg_text_lines(lines: list[str], x: int, y: int, size: int, fill: str) -> str:
    chunks = []
    for index, line in enumerate(lines):
        chunks.append(
            f'<text x="{x}" y="{y + index * int(size * 1.35)}" font-size="{size}" fill="{fill}" font-family="Arial, sans-serif">{html.escape(line)}</text>'
        )
    return "\n".join(chunks)


def create_svg(scene: dict[str, object], path: Path) -> None:
    title_lines = wrap_text(str(scene["title"]), 30)
    copy_lines = wrap_text(str(scene["copy"]), 44)
    prompt_lines = wrap_text(str(scene["visual_prompt"]), 58)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="675" viewBox="0 0 1200 675">
  <rect width="1200" height="675" fill="{scene['background']}"/>
  <rect x="64" y="64" width="1072" height="547" rx="24" fill="#ffffff" opacity="0.86"/>
  <circle cx="1000" cy="154" r="74" fill="{scene['accent']}" opacity="0.18"/>
  <circle cx="964" cy="204" r="42" fill="{scene['accent']}" opacity="0.28"/>
  <rect x="96" y="110" width="236" height="156" rx="18" fill="{scene['accent']}" opacity="0.16"/>
  <rect x="126" y="142" width="176" height="18" rx="9" fill="{scene['accent']}" opacity="0.62"/>
  <rect x="126" y="178" width="132" height="18" rx="9" fill="{scene['accent']}" opacity="0.42"/>
  <rect x="126" y="214" width="154" height="18" rx="9" fill="{scene['accent']}" opacity="0.42"/>
  <path d="M370 188 C470 104, 606 104, 706 188 S940 272, 1040 188" fill="none" stroke="{scene['accent']}" stroke-width="12" stroke-linecap="round" opacity="0.48"/>
  <rect x="762" y="112" width="256" height="176" rx="22" fill="{scene['accent']}" opacity="0.12"/>
  <path d="M806 230 L864 170 L916 222 L946 194 L990 244" fill="none" stroke="{scene['accent']}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round" opacity="0.68"/>
  {svg_text_lines(title_lines, 96, 352, 42, scene['text_color'])}
  {svg_text_lines(copy_lines, 96, 458, 28, scene['text_color'])}
  <rect x="96" y="548" width="1008" height="38" rx="19" fill="#111827" opacity="0.06"/>
  {svg_text_lines(prompt_lines, 118, 574, 18, '#374151')}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def build_prompt_plan(brief: dict[str, object], scenes: list[dict[str, object]]) -> list[dict[str, object]]:
    plan = []
    for index, scene in enumerate(scenes, start=1):
        plan.append({
            "version": f"v{index}",
            "scene_id": scene["id"],
            "task": "image_generation_prompt",
            "prompt": scene["visual_prompt"],
            "negative_prompt": "no real person likeness, no brand logo, no copyrighted character, no unreadable text",
            "size": "1200x675",
            "style": brief["tone"],
            "status": "ready_for_model_or_svg_baseline",
        })
    return plan


def review_scene(scene: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    contrast = contrast_ratio(str(scene["background"]), str(scene["text_color"]))
    checks = {
        "source_recorded": bool(scene.get("source")),
        "license_recorded": bool(scene.get("license")),
        "no_real_person_likeness": not bool(scene.get("uses_real_person")),
        "sufficient_contrast": contrast >= 4.5,
        "export_limits_written": True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "scene_id": scene["id"],
        "title": scene["title"],
        "source": scene["source"],
        "license": scene["license"],
        "contrast_ratio": round(contrast, 2),
        "passed": not failures,
        **checks,
    }, failures


def build_storyboard(scenes: list[dict[str, object]]) -> list[dict[str, object]]:
    elapsed = 0
    storyboard = []
    for scene in scenes:
        start = elapsed
        end = start + int(scene["duration_sec"])
        elapsed = end
        storyboard.append({
            "scene_id": scene["id"],
            "start_sec": start,
            "end_sec": end,
            "visual": scene["visual_prompt"],
            "voiceover": scene["copy"],
            "asset_file": f"assets/{scene['id']}.svg",
        })
    return storyboard


def build_html_preview(brief: dict[str, object], scenes: list[dict[str, object]], review_rows: list[dict[str, object]]) -> str:
    cards = []
    for scene, review in zip(scenes, review_rows):
        status = "通过" if review["passed"] else "需复核"
        cards.append(f'''
<section class="card">
  <img src="../assets/{scene['id']}.svg" alt="{html.escape(str(scene['title']))}">
  <h2>{html.escape(str(scene['title']))}</h2>
  <p>{html.escape(str(scene['copy']))}</p>
  <p><strong>审核：</strong> {status} | 对比度 {review['contrast_ratio']}</p>
</section>''')
    return f'''<!doctype html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8">
  <title>{html.escape(str(brief['topic']))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f8fafc; color: #111827; }}
    main {{ max-width: 960px; margin: 0 auto; padding: 32px; }}
    .card {{ background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
    img {{ width: 100%; border-radius: 6px; border: 1px solid #e5e7eb; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(str(brief['topic']))}</h1>
  <p>{html.escape(str(brief['goal']))}</p>
  {''.join(cards)}
</main>
</body>
</html>
'''


def main() -> None:
    write_json(INPUT_DIR / "creative_brief.json", BRIEF)
    prompt_plan = build_prompt_plan(BRIEF, SCENES)
    write_json(PROMPT_DIR / "prompt_plan.json", prompt_plan)
    prompt_md = ["# Prompt Versions", ""]
    for item in prompt_plan:
        prompt_md.append(f"## {item['version']} - {item['scene_id']}")
        prompt_md.append(f"- Prompt: {item['prompt']}")
        prompt_md.append(f"- Negative prompt: {item['negative_prompt']}")
        prompt_md.append(f"- Status: {item['status']}")
        prompt_md.append("")
    (PROMPT_DIR / "prompt_versions.md").write_text("\n".join(prompt_md), encoding="utf-8")

    review_rows = []
    failure_cases = []
    for scene in SCENES:
        asset_path = ASSET_DIR / f"{scene['id']}.svg"
        create_svg(scene, asset_path)
        review, failures = review_scene(scene)
        review_rows.append(review)
        if failures:
            failure_cases.append({
                "scene_id": scene["id"],
                "title": scene["title"],
                "failures": failures,
                "suspected_cause": "visual design or missing asset metadata does not meet export rules",
                "fix_action": "adjust colors, source records, license notes, or review thresholds, then rerun the script",
            })

    write_csv(REPORT_DIR / "asset_manifest.csv", review_rows, ["scene_id", "title", "source", "license", "contrast_ratio", "passed", *REVIEW_RULES])
    storyboard = build_storyboard(SCENES)
    write_json(OUTPUT_DIR / "storyboard.json", storyboard)
    write_csv(OUTPUT_DIR / "timeline.csv", storyboard, ["scene_id", "start_sec", "end_sec", "visual", "voiceover", "asset_file"])
    write_json(OUTPUT_DIR / "content_package.json", {"brief": BRIEF, "prompts": prompt_plan, "storyboard": storyboard, "review": review_rows})
    (OUTPUT_DIR / "export_preview.html").write_text(build_html_preview(BRIEF, SCENES, review_rows), encoding="utf-8")

    safety_lines = ["# Safety Review", "", "| Scene | Source | License | Contrast | Result |", "|---|---|---|---:|---|"]
    for row in review_rows:
        result = "PASS" if row["passed"] else "NEEDS REVIEW"
        safety_lines.append(f"| {row['scene_id']} | {row['source']} | {row['license']} | {row['contrast_ratio']} | {result} |")
    safety_lines.append("")
    safety_lines.append("导出限制：仅用于课程演示；公开发布前，请用经过复核的模型输出替换 SVG 基线素材。")
    (REPORT_DIR / "safety_review.md").write_text("\n".join(safety_lines), encoding="utf-8")

    failure_lines = ["# 失败案例", ""]
    if not failure_cases:
        failure_lines.append("没有触发失败案例。在把它作为作品集报告之前，请加入边界样本。")
    for index, case in enumerate(failure_cases, start=1):
        failure_lines.append(f"## 案例 {index}：{case['scene_id']}")
        failure_lines.append(f"- 标题：{case['title']}")
        failure_lines.append(f"- 失败项：{', '.join(case['failures'])}")
        failure_lines.append(f"- 疑似原因：{case['suspected_cause']}")
        failure_lines.append(f"- 修复动作：{case['fix_action']}")
        failure_lines.append("")
    (REPORT_DIR / "failure_cases.md").write_text("\n".join(failure_lines), encoding="utf-8")

    readme = f"""# 多模态工作坊运行

运行命令：

~~~bash
python multimodal_workshop.py
~~~

产物：

- inputs/creative_brief.json
- prompts/prompt_plan.json and prompts/prompt_versions.md
- assets/*.svg
- outputs/storyboard.json
- outputs/export_preview.html
- reports/asset_manifest.csv
- reports/safety_review.md
- reports/failure_cases.md

Summary:

- scenes: {len(SCENES)}
- generated_svg_assets: {len(list(ASSET_DIR.glob('*.svg')))}
- review_passed: {sum(1 for row in review_rows if row['passed'])}/{len(review_rows)}
- failure_cases: {len(failure_cases)}
"""
    (RUN_DIR / "README.md").write_text(readme, encoding="utf-8")

    print("STEP 1: project brief")
    print(f"topic: {BRIEF['topic']}")
    print(f"deliverables: {len(BRIEF['deliverables'])}")
    print("")
    print("STEP 2: generated assets")
    print(f"svg_assets: {len(list(ASSET_DIR.glob('*.svg')))}")
    print(f"storyboard_scenes: {len(storyboard)}")
    print(f"review_passed: {sum(1 for row in review_rows if row['passed'])}/{len(review_rows)}")
    print(f"failure_cases: {len(failure_cases)}")
    print("")
    print("STEP 3: files to inspect")
    print(f"prompt_plan: {PROMPT_DIR / 'prompt_plan.json'}")
    print(f"asset_manifest: {REPORT_DIR / 'asset_manifest.csv'}")
    print(f"safety_review: {REPORT_DIR / 'safety_review.md'}")
    print(f"export_preview: {OUTPUT_DIR / 'export_preview.html'}")


if __name__ == "__main__":
    main()
```

运行：

```bash
python multimodal_workshop.py
```

预期输出：

```text
STEP 1: project brief
topic: AI learning assistant launch kit
deliverables: 4

STEP 2: generated assets
svg_assets: 3
storyboard_scenes: 3
review_passed: 2/3
failure_cases: 1

STEP 3: files to inspect
prompt_plan: multimodal_workshop_run/prompts/prompt_plan.json
asset_manifest: multimodal_workshop_run/reports/asset_manifest.csv
safety_review: multimodal_workshop_run/reports/safety_review.md
export_preview: multimodal_workshop_run/outputs/export_preview.html
```

![workshop 运行结果证据包图](/img/course/ch12-workshop-run-evidence-package-result-map.webp)

## 步骤 3：检查 Brief 和 Prompt 记录

打开 `inputs/creative_brief.json`。这是结构化后的用户需求：主题、受众、目标、语气、交付物和约束。

再打开 `prompts/prompt_plan.json` 和 `prompts/prompt_versions.md`。真实 AIGC 项目不能在图片或视频生成后丢掉 Prompt。Prompt 本身就是项目证据的一部分。

## 步骤 4：检查资产和分镜

用浏览器打开 `assets/scene_01.svg`、`assets/scene_02.svg` 和 `assets/scene_03.svg`。它们是脚本生成的基线 SVG 素材，在工作流里扮演“生成结果”的角色。

打开 `outputs/storyboard.json` 和 `outputs/timeline.csv`。这些文件说明视觉资产如何组成短视频或落地页顺序。

![审核与导出流程图](/img/course/ch12-workshop-review-export-map.webp)

## 步骤 5：阅读审核文件

打开 `reports/asset_manifest.csv`。每一行记录：

| 字段 | 含义 |
|---|---|
| `source` | 资产来源 |
| `license` | 资产是否能用于这个演示 |
| `contrast_ratio` | 文本是否足够可读 |
| `passed` | 是否可以进入导出 |

然后打开 `reports/safety_review.md`。这个文件用于记录版权、人像权、内容安全和导出边界。

## 步骤 6：打开导出预览

用浏览器打开 `outputs/export_preview.html`。它不是完整应用，但能证明项目已经从创意需求走到可导出的内容包。

真实升级时，可以替换这些模块：

| 基线模块 | 真实项目替换 |
|---|---|
| 基线 SVG 素材 | 图像生成 API 或本地图像模型 |
| Storyboard JSON | 视频生成工作流 |
| 文案文本 | LLM 生成文案加人工审核 |
| 安全检查表 | 人工审核加策略检查 |
| HTML preview | 前端创意工作台 |

## 步骤 7：阅读失败报告

![多模态失败样本排查图](/img/course/ch12-workshop-failure-debug-map.webp)

打开 `reports/failure_cases.md`。这个练习故意让一个 scene 没通过对比度检查。这样做有用，因为作品集项目要展示你如何发现问题，而不是只展示好看的输出。

每个失败案例都问：

1. 素材来源是否记录？
2. 授权或使用范围是否清楚？
3. 资产是否有人像或品牌风险？
4. 生成内容是否可读、可导出？
5. 项目是否写明发布前必须人工确认的内容？

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
简介：用户目标、受众、素材、约束和导出格式
工件：源文件、提示词、生成候选、选定输出和被拒绝版本
审查：事实检查、版权/肖像/敏感内容检查，以及人工决定
集成: RAG 记录、Agent trace、创意包、故事板或导出预览
期望产出：可复现的资产包，包含 README、复查清单和失败说明
```

## 常见错误

| 现象 | 可能原因 | 修复方向 |
|---|---|---|
| 生成资产无法复用 | 没有来源或授权记录 | 增加 `asset_manifest.csv`，逐个审核资产 |
| Prompt 版本丢失 | Prompt 只留在聊天记录里 | 生成前保存 `prompt_plan.json` |
| 视频脚本不连贯 | 没有 storyboard 或 timeline | 先写 `storyboard.json` 再生成 |
| 结果好看但不能发布 | 没有版权、人像或安全审核 | 增加 `safety_review.md` 和导出限制 |
| 用户不能比较版本 | 资产互相覆盖 | 增加 scene id、prompt version 和输出目录 |

## 练习任务

1. 修复 `scene_02` 配色，让所有场景通过对比度审核。
2. 新增 `scene_04`，并扩展 storyboard timeline。
3. 给 safety review 增加 `manual_reviewer` 字段。
4. 用你喜欢的图像模型生成一张图替换某个 SVG，但保留同样的 manifest 和 review 文件。
5. 新增一个故意有风险的资产，确认它进入 `failure_cases.md`。

<details>
<summary>操作参考与检查点</summary>

1. `scene_02` 通过的标准是前景/背景对比度满足其他场景使用的同一条评审规则，并且 manifest 记录了这次修改。
2. `scene_04` 应加入 storyboard，包含 id、用途、时长/顺序、所需资产、prompt 版本和评审结果，这样时间线才可复现。
3. `manual_reviewer` 应写入安全评审记录，而不是只放在 README 备注里。它应说明是谁评审，或哪个角色批准了资产。
4. 替换 SVG 可以，但生成图必须保留同一份 manifest 入口、来源/prompt 记录、选中输出路径和评审文件。
5. 故意加入的高风险资产不应静默通过。好的结果是它进入 `failure_cases.md`，并写清原因、风险类别和下一步处理。

</details>
## 完成标准

完成本练习后，你应该能解释：

- creative brief 控制什么；
- 为什么 Prompt 和资产需要版本记录；
- storyboard 如何把资产组织成视频或页面顺序；
- 为什么安全审核是工作流的一部分，而不是最后补丁；
- 哪些文件能证明项目可复现。

这是第 12 章最小但有用的 baseline：一个能生成、审核、导出并讲清楚的多模态项目。
