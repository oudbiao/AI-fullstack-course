#!/usr/bin/env python3
"""Render deterministic ch03 Python database workflow images.

The page uses exact table names, field names, and query patterns. Rendering the
diagram locally keeps multilingual text readable and avoids invented metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W = 1024
H = 1792

FONT_PATHS = [
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/ArialHB.ttc",
]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            for index in ([1, 0, 2] if bold else [0, 1, 2]):
                try:
                    return ImageFont.truetype(path, size=size, index=index)
                except OSError:
                    continue
    return ImageFont.load_default()


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0]


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return []
    if " " in text:
        lines: list[str] = []
        current = ""
        for token in text.split(" "):
            candidate = token if not current else f"{current} {token}"
            if text_width(draw, candidate, fnt) <= max_width:
                current = candidate
                continue
            if current:
                lines.append(current)
            current = token
        if current:
            lines.append(current)
        return lines

    lines: list[str] = []
    current = ""
    for char in text:
        candidate = current + char
        if not current or text_width(draw, candidate, fnt) <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = char
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    *,
    fill: str,
    max_width: int,
    gap: int = 6,
) -> int:
    x, y = xy
    for line in wrap(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += getattr(fnt, "size", 24) + gap
    return y


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str,
    *,
    width: int = 2,
    radius: int = 18,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def pill(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, *, fill: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    pad_x = 14
    pad_y = 7
    box_w = text_width(draw, text, fnt) + pad_x * 2
    box_h = getattr(fnt, "size", 20) + pad_y * 2
    rounded(draw, (x, y, x + box_w, y + box_h), fill, fill, radius=12)
    draw.text((x + pad_x, y + pad_y - 1), text, font=fnt, fill="#ffffff")
    return x + box_w, y + box_h


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], *, color: str = "#38bdf8") -> None:
    sx, sy = start
    ex, ey = end
    draw.line((sx, sy, ex, ey), fill=color, width=5)
    if ex >= sx:
        head = [(ex, ey), (ex - 18, ey - 10), (ex - 18, ey + 10)]
    else:
        head = [(ex, ey), (ex + 18, ey - 10), (ex + 18, ey + 10)]
    draw.polygon(head, fill=color)


def code_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], lines: list[str], *, title: str | None = None) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, "#08111f", "#38bdf8", width=2, radius=12)
    y = y1 + 14
    if title:
        draw.text((x1 + 16, y), title, font=font(23, bold=True), fill="#bae6fd")
        y += 36
        draw.line((x1 + 14, y, x2 - 14, y), fill="#1e3a5f", width=2)
        y += 12
    available = y2 - y - 10
    line_h = max(24, min(30, available // max(1, len(lines))))
    fnt = font(18 if line_h < 28 else 19)
    for line in lines:
        if y + getattr(fnt, "size", 20) > y2 - 8:
            break
        draw.text((x1 + 16, y), line, font=fnt, fill="#e2e8f0")
        y += line_h


def table_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    fields: list[str],
    *,
    accent: str = "#22c55e",
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, "#0f172a", accent, width=3, radius=14)
    draw.rounded_rectangle((x1, y1, x2, y1 + 48), radius=14, fill="#17324a", outline=accent, width=3)
    draw.text((x1 + 16, y1 + 10), title, font=font(25, bold=True), fill="#f8fafc")
    y = y1 + 58
    row_h = max(22, min(38, (y2 - y - 8) // max(1, len(fields))))
    fnt = font(17 if row_h < 28 else 19 if row_h < 32 else 21)
    for field in fields:
        if y + getattr(fnt, "size", 20) > y2 - 6:
            break
        draw.line((x1 + 10, y - 8, x2 - 10, y - 8), fill="#334155", width=1)
        draw.text((x1 + 18, y), field, font=fnt, fill="#e2e8f0")
        y += row_h


def step_panel(
    draw: ImageDraw.ImageDraw,
    y: int,
    number: str,
    title: str,
    subtitle: str,
    *,
    height: int,
    accent: str,
) -> tuple[int, int, int, int]:
    x1, x2 = 36, W - 36
    box = (x1, y, x2, y + height)
    rounded(draw, box, "#f8fafc", "#cbd5e1", width=2, radius=22)
    draw.rounded_rectangle((x1 + 18, y + 18, x1 + 80, y + 80), radius=14, fill=accent, outline=accent)
    draw.text((x1 + 38, y + 25), number, font=font(36, bold=True), fill="#ffffff")
    draw.text((x1 + 100, y + 20), title, font=font(34, bold=True), fill="#0f172a")
    draw_wrapped(draw, (x1 + 102, y + 62), subtitle, font(21), fill="#475569", max_width=790, gap=4)
    return box


TEXT = {
    "zh": {
        "file": "ch03-python-database-safety-vertical.png",
        "title": "Python 数据库安全协作闭环",
        "subtitle": "tickets 表 -> 参数化 SQL -> Pandas 汇总 -> ticket_summary 写回",
        "steps": [
            ("1", "连接与建表", "先把应用问题落到 tickets 表。字段少而清楚，后续查询才稳定。"),
            ("2", "安全查询", "外部输入只通过占位符传入，避免 SQL 注入。"),
            ("3", "交给 Pandas", "SQL 先筛选行，Pandas 再做分组、统计和看板数据。"),
            ("4", "写回结果", "只把干净的汇总表写回数据库，保留可复用证据。"),
        ],
        "question": "应用要回答：哪些 open 工单需要优先处理？",
        "bad": "错误：f-string 拼接",
        "good": "正确：参数化查询",
        "sql_title": "安全 SQL",
        "pandas_title": "Pandas 汇总",
        "write_title": "写回数据库",
        "footer": "把这张证据卡放进项目 README：它证明你会安全查询、分析并写回结果。",
    },
    "en": {
        "file": "ch03-python-database-safety-vertical-en.png",
        "title": "Safe Python Database Workflow",
        "subtitle": "tickets table -> parameterized SQL -> Pandas summary -> ticket_summary write-back",
        "steps": [
            ("1", "Connect and Model", "Turn the application question into a clear tickets table."),
            ("2", "Safe Query", "Pass external input through placeholders to avoid SQL injection."),
            ("3", "Hand Off to Pandas", "Filter rows with SQL first, then group and summarize in Pandas."),
            ("4", "Write Back", "Store only the clean summary table you need to reuse."),
        ],
        "question": "Application question: which open tickets need attention first?",
        "bad": "Wrong: f-string SQL",
        "good": "Right: parameterized query",
        "sql_title": "Safe SQL",
        "pandas_title": "Pandas Summary",
        "write_title": "Write Back",
        "footer": "Keep this evidence card in the project README: it proves safe query, analysis, and write-back.",
    },
    "ja": {
        "file": "ch03-python-database-safety-vertical-ja.png",
        "title": "安全な Python データベース連携",
        "subtitle": "tickets 表 -> パラメータ化 SQL -> Pandas 集計 -> ticket_summary 書き戻し",
        "steps": [
            ("1", "接続とモデル化", "アプリの問いを、明確な tickets 表に落とし込みます。"),
            ("2", "安全なクエリ", "外部入力はプレースホルダーで渡し、SQL インジェクションを避けます。"),
            ("3", "Pandas に渡す", "まず SQL で行を絞り、その後 Pandas で集計します。"),
            ("4", "結果を書き戻す", "再利用するために、必要な集計表だけを保存します。"),
        ],
        "question": "アプリの問い：どの open チケットを優先すべきか？",
        "bad": "悪い例：f-string SQL",
        "good": "良い例：パラメータ化クエリ",
        "sql_title": "安全な SQL",
        "pandas_title": "Pandas 集計",
        "write_title": "書き戻し",
        "footer": "この証拠カードを README に残し、安全な検索、分析、書き戻しを示します。",
    },
}


def render(locale: str, out_dir: Path) -> Path:
    t = TEXT[locale]
    img = Image.new("RGB", (W, H), "#eaf2f8")
    draw = ImageDraw.Draw(img)

    # Background bands.
    draw.rectangle((0, 0, W, 210), fill="#0f2a44")
    draw.rectangle((0, 210, W, H), fill="#edf6fb")
    draw.rectangle((0, 1618, W, H), fill="#0f2a44")

    draw.text((42, 34), t["title"], font=font(45, bold=True), fill="#ffffff")
    draw_wrapped(draw, (44, 92), t["subtitle"], font(24), fill="#dbeafe", max_width=930)
    x, _ = pill(draw, 44, 148, "sqlite3", fill="#2563eb", fnt=font(22, bold=True))
    x, _ = pill(draw, x + 12, 148, "parameterized SQL", fill="#0891b2", fnt=font(22, bold=True))
    pill(draw, x + 12, 148, "Pandas", fill="#16a34a", fnt=font(22, bold=True))

    panels = [
        (232, 330, "#2563eb"),
        (586, 320, "#dc2626"),
        (930, 340, "#16a34a"),
        (1294, 300, "#7c3aed"),
    ]

    for (index, (num, title, subtitle)) in enumerate(t["steps"]):
        y, height, accent = panels[index]
        step_panel(draw, y, num, title, subtitle, height=height, accent=accent)

    # Panel 1 content.
    table_box(
        draw,
        (72, 344, 426, 532),
        "tickets",
        ["id PK", "customer", "status", "priority", "first_reply_minutes"],
        accent="#2563eb",
    )
    draw_wrapped(draw, (468, 360), t["question"], font(27, bold=True), fill="#0f172a", max_width=450)
    code_box(
        draw,
        (470, 432, 924, 544),
        ["SELECT * FROM tickets", "WHERE status = ?"],
        title=t["sql_title"],
    )

    # Panel 2 content.
    code_box(
        draw,
        (72, 704, 456, 846),
        ["sql = f\"SELECT * FROM tickets\"", "\"WHERE customer = '{user_input}'\""],
        title=t["bad"],
    )
    draw.line((88, 858, 438, 858), fill="#ef4444", width=5)
    draw.text((92, 868), "SQL injection risk", font=font(24, bold=True), fill="#b91c1c")
    code_box(
        draw,
        (510, 704, 924, 868),
        ["cursor.execute(", "  \"SELECT * FROM tickets", "   WHERE customer = ?\",", "  (user_input,))"],
        title=t["good"],
    )
    arrow(draw, (456, 776), (510, 776), color="#64748b")

    # Panel 3 content.
    code_box(
        draw,
        (72, 1060, 438, 1202),
        ["SELECT status, priority,", "       first_reply_minutes", "FROM tickets", "WHERE status != 'closed'"],
        title=t["sql_title"],
    )
    arrow(draw, (438, 1132), (502, 1132), color="#16a34a")
    table_box(
        draw,
        (502, 1060, 924, 1212),
        "DataFrame",
        ["status", "priority", "first_reply_minutes"],
        accent="#16a34a",
    )
    rounded(draw, (162, 1228, 862, 1266), "#08111f", "#38bdf8", width=2, radius=12)
    draw.text((178, 1236), t["pandas_title"], font=font(22, bold=True), fill="#bae6fd")
    draw.text((430, 1237), "groupby summary", font=font(18), fill="#e2e8f0")

    # Panel 4 content.
    table_box(
        draw,
        (88, 1424, 458, 1586),
        "ticket_summary",
        ["status", "priority", "ticket_count", "avg_first_reply"],
        accent="#7c3aed",
    )
    arrow(draw, (458, 1488), (542, 1488), color="#7c3aed")
    code_box(
        draw,
        (542, 1424, 920, 1586),
        ["summary.to_sql(", "  'ticket_summary', conn,", "  if_exists='replace',", "  index=False", ")"],
        title=t["write_title"],
    )

    draw.text((42, 1640), "Evidence card", font=font(28, bold=True), fill="#ffffff")
    draw_wrapped(
        draw,
        (42, 1684),
        "schema: tickets | query: parameterized SQL | output: ticket_summary | failure_check: unsafe query or missing commit",
        font(23),
        fill="#dbeafe",
        max_width=930,
    )
    draw_wrapped(draw, (42, 1740), t["footer"], font(18), fill="#bfdbfe", max_width=930)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / t["file"]
    img.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="tmp/ch03-python-db-content-sync-20260521")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    for locale in ("zh", "en", "ja"):
        print(render(locale, out_dir))


if __name__ == "__main__":
    main()
