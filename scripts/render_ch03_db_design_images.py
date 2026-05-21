#!/usr/bin/env python3
"""Render deterministic ch03 database-design teaching images.

These images contain field names that must be exact, so they are rendered
locally instead of relying on image-model text generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


W = 1024
H = 1792


FONT_PATHS = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=1 if bold else 0)
            except OSError:
                try:
                    return ImageFont.truetype(path, size=size)
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


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    *,
    fill: str = "#f8fafc",
    max_width: int | None = None,
    line_gap: int = 4,
) -> int:
    x, y = xy
    lines = [text] if max_width is None else wrap(draw, text, fnt, max_width)
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + line_gap if hasattr(fnt, "size") else 24
    return y


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str, outline: str, width: int = 2, radius: int = 12) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color: str, fnt: ImageFont.ImageFont) -> int:
    pad_x = 16
    pad_y = 8
    w = text_width(draw, text, fnt) + pad_x * 2
    h = fnt.size + pad_y * 2
    rounded(draw, (x, y, x + w, y + h), color, color, radius=8)
    draw.text((x + pad_x, y + pad_y - 1), text, font=fnt, fill="#ffffff")
    return x + w + 12


def draw_cross(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 20) -> None:
    draw.line((x, y, x + size, y + size), fill="#f87171", width=4)
    draw.line((x + size, y, x, y + size), fill="#f87171", width=4)


def draw_tick(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 22) -> None:
    draw.line((x, y + size // 2, x + size // 3, y + size), fill="#86efac", width=4)
    draw.line((x + size // 3, y + size, x + size, y), fill="#86efac", width=4)


def section_line(draw: ImageDraw.ImageDraw, y: int) -> None:
    draw.line((24, y, W - 24, y), fill="#334155", width=3)


def draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    title: str,
    fields: list[str],
    *,
    accent: str,
    fill: str = "#0f172a",
    title_fill: str = "#17324a",
    row_h: int = 34,
    title_h: int = 42,
) -> tuple[int, int, int, int]:
    header_font = font(21, bold=True)
    field_font = font(18)
    h = title_h + row_h * len(fields)
    rounded(draw, (x, y, x + w, y + h), fill, accent, radius=8)
    draw.rounded_rectangle((x, y, x + w, y + title_h), radius=8, fill=title_fill, outline=accent, width=2)
    draw.text((x + 12, y + 8), title, font=header_font, fill="#f8fafc")
    yy = y + title_h
    for field in fields:
        draw.line((x, yy, x + w, yy), fill="#475569", width=1)
        draw.text((x + 12, yy + 7), field, font=field_font, fill="#e2e8f0")
        yy += row_h
    return (x, y, x + w, y + h)


def connect(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#93c5fd") -> None:
    sx, sy = start
    ex, ey = end
    mid = (sx + ex) // 2
    draw.line((sx, sy, mid, sy, mid, ey, ex, ey), fill=color, width=3)
    r = 4
    draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=color)


def connect_path(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str = "#93c5fd") -> None:
    flattened = [coord for point in points for coord in point]
    draw.line(flattened, fill=color, width=3, joint="curve")
    ex, ey = points[-1]
    r = 4
    draw.ellipse((ex - r, ey - r, ex + r, ey + r), fill=color)


LOCALES = {
    "zh": {
        "file": "ch03-database-design-erd-normalization.png",
        "locale": "zh",
        "title": "先分表，再连表，再补索引",
        "subtitle": "范式不是背概念，而是减少重复、更新冲突和维护事故。",
        "bad": "坏宽表",
        "bad_note": "同一事实反复出现，改一处不等于都改对。",
        "warn_1": "1NF 警告",
        "warn_1_note": "customer_phone 里塞了多个值。",
        "warn_2": "重复事实",
        "warn_2_note": "客户、商品、供应商信息被反复复制。",
        "split": "分表",
        "split_note": "按实体拆成客户、订单、商品。",
        "key": "主键 / 外键",
        "key_note": "用稳定 ID 连接关系。",
        "ticket_title": "客服工单系统 ERD",
        "index_title": "索引查询路径",
        "index_note": "索引要服务真实查询。",
        "indexes": [
            "按 assignee_id 查未解决工单",
            "按 customer_id 查客户工单",
            "按 ticket_id 查工单消息",
            "按 status 查状态",
            "按 category_id 查分类",
            "按 tag 查标签",
        ],
        "check": "检查",
        "before": "分表前：坏宽表风险",
        "after": "分表后：规范化 + 索引",
        "before_rows": ["重复多", "更新容易冲突", "查询路径不清"],
        "after_rows": ["重复少", "更新稳定", "查询更快"],
        "footer": "好的数据库设计，是为了以后少出错、好维护。",
    },
    "en": {
        "file": "ch03-database-design-erd-normalization-en.png",
        "locale": "en",
        "title": "Split Tables, Link Them, Then Add Indexes",
        "subtitle": "Normalization means fewer duplicates, safer updates, and easier maintenance.",
        "bad": "Bad wide table",
        "bad_note": "One fact is copied many times, so edits can drift.",
        "warn_1": "1NF warning",
        "warn_1_note": "customer_phone stores multiple values.",
        "warn_2": "Duplicated facts",
        "warn_2_note": "Customer, product, and supplier facts repeat.",
        "split": "Split tables",
        "split_note": "Separate customers, orders, and products.",
        "key": "Primary / foreign key",
        "key_note": "Link relations with stable IDs.",
        "ticket_title": "Support ticket system ERD",
        "index_title": "Index query paths",
        "index_note": "Indexes should match real queries.",
        "indexes": [
            "assignee_id: open tickets",
            "customer_id: customer tickets",
            "ticket_id: ticket messages",
            "status: ticket status",
            "category_id: category totals",
            "tag: tagged tickets",
        ],
        "check": "Check",
        "before": "Before: bad wide-table risks",
        "after": "After: normalized + indexed",
        "before_rows": ["Many duplicates", "Updates touch many rows", "Lookup paths are unclear"],
        "after_rows": ["Minimal duplicates", "Update in one place", "Indexed lookups are fast"],
        "footer": "Good database design makes future changes safer and easier.",
    },
    "ja": {
        "file": "ch03-database-design-erd-normalization-ja.png",
        "locale": "ja",
        "title": "表を分け、つなぎ、索引を足す",
        "subtitle": "正規化は暗記ではなく、重複と保守事故を減らすため。",
        "bad": "悪い wide table",
        "bad_note": "同じ事実が何度も出ると、更新でずれやすい。",
        "warn_1": "1NF 警告",
        "warn_1_note": "customer_phone に複数値が入っている。",
        "warn_2": "重複した事実",
        "warn_2_note": "顧客、商品、仕入先情報が繰り返される。",
        "split": "分割",
        "split_note": "顧客、注文、商品に分ける。",
        "key": "primary / foreign key",
        "key_note": "安定した ID で関係をつなぐ。",
        "ticket_title": "サポートチケット ERD",
        "index_title": "索引の検索経路",
        "index_note": "索引は実際の検索に合わせる。",
        "indexes": [
            "assignee_id で未解決チケットを探す",
            "customer_id で顧客チケットを探す",
            "ticket_id でチケットメッセージを探す",
            "status で状態を絞る",
            "category_id で分類を集計する",
            "tag でタグ付きチケットを探す",
        ],
        "check": "確認",
        "before": "分割前：wide table のリスク",
        "after": "分割後：正規化 + 索引",
        "before_rows": ["重複が多い", "更新が衝突しやすい", "検索経路が曖昧"],
        "after_rows": ["重複が少ない", "一箇所で更新", "索引で検索が速い"],
        "footer": "よい DB 設計は、後の変更を安全で楽にする。",
    },
}


def background(draw: ImageDraw.ImageDraw) -> None:
    for y in range(H):
        shade = int(15 + y / H * 12)
        draw.line((0, y, W, y), fill=(shade, shade + 9, shade + 14))
    for x in range(0, W, 64):
        draw.line((x, 0, x, H), fill="#172033", width=1)
    for y in range(0, H, 64):
        draw.line((0, y, W, y), fill="#172033", width=1)


def draw_wide_table(draw: ImageDraw.ImageDraw, data: dict[str, object]) -> None:
    y = 190
    note_x = badge(draw, 36, y, str(data["bad"]), "#b91c1c", font(27, bold=True))
    draw_text(draw, (note_x + 4, y + 3), str(data["bad_note"]), font(24, bold=True), fill="#fde68a", max_width=520)

    x = 36
    table_y = y + 64
    table_w = 712
    columns = ["customer_name", "customer_phone", "product_name", "supplier", "quantity"]
    rows = [
        ["CUST_A", "PHONE_A1, PHONE_A2", "PROD_A", "SUP_A", "QTY"],
        ["CUST_A", "PHONE_A1, PHONE_A2", "PROD_B", "SUP_B", "QTY"],
        ["CUST_B", "PHONE_B1, PHONE_B2", "PROD_A", "SUP_A", "QTY"],
    ]
    col_ws = [145, 205, 145, 105, 112]
    title_h = 38
    header_h = 42
    row_h = 54
    h = title_h + header_h + row_h * len(rows)
    rounded(draw, (x, table_y, x + table_w, table_y + h), "#111827", "#f87171", width=3, radius=8)
    draw.rectangle((x, table_y, x + table_w, table_y + title_h), fill="#3b1717")
    title_font = font(24, bold=True)
    draw.text((x + 18, table_y + 7), "order_line_snapshot", font=title_font, fill="#fecaca")
    xx = x
    header_font = font(17, bold=True)
    body_font = font(17)
    compact_font = font(15)
    for col, cw in zip(columns, col_ws):
        draw.rectangle((xx, table_y + title_h, xx + cw, table_y + title_h + header_h), fill="#1f2937", outline="#64748b")
        draw.text((xx + 8, table_y + title_h + 12), col, font=header_font, fill="#e5e7eb")
        xx += cw
    yy = table_y + title_h + header_h
    for row in rows:
        xx = x
        for value, cw in zip(row, col_ws):
            draw.rectangle((xx, yy, xx + cw, yy + row_h), outline="#475569")
            if "," in value:
                first, second = [part.strip() for part in value.split(",", 1)]
                draw.text((xx + 8, yy + 9), first + ",", font=compact_font, fill="#f8fafc")
                draw.text((xx + 8, yy + 29), second, font=compact_font, fill="#f8fafc")
            else:
                draw.text((xx + 8, yy + 17), value, font=body_font, fill="#f8fafc")
            xx += cw
        yy += row_h

    warning_x = 780
    warning_font = font(21, bold=True)
    note_font = font(18)
    for i, (title, note) in enumerate(
        [
            (str(data["warn_1"]), str(data["warn_1_note"])),
            (str(data["warn_2"]), str(data["warn_2_note"])),
        ]
    ):
        box_y = table_y + 6 + i * 116
        rounded(draw, (warning_x, box_y, 984, box_y + 108), "#261818", "#ef4444", width=2, radius=10)
        draw.text((warning_x + 18, box_y + 14), "!", font=font(34, bold=True), fill="#f87171")
        draw.text((warning_x + 52, box_y + 18), title, font=warning_font, fill="#fca5a5")
        draw_text(draw, (warning_x + 18, box_y + 52), note, note_font, fill="#f8fafc", max_width=176, line_gap=1)
    section_line(draw, 520)


def draw_split_tables(draw: ImageDraw.ImageDraw, data: dict[str, object]) -> None:
    y = 544
    x = badge(draw, 36, y, str(data["split"]), "#15803d", font(28, bold=True))
    draw_text(draw, (x + 4, y + 5), str(data["split_note"]), font(25, bold=True), fill="#dcfce7", max_width=580)
    badge(draw, 36, y + 276, str(data["key"]), "#0f766e", font(22, bold=True))
    draw_text(draw, (36, y + 332), str(data["key_note"]), font(21), fill="#ccfbf1", max_width=230)

    tables = {
        "customers": draw_table(draw, 290, y + 66, 178, "customers", ["customer_id PK", "customer_name"], accent="#86efac", title_fill="#174331"),
        "customer_phones": draw_table(draw, 540, y + 54, 205, "customer_phones", ["phone_id PK", "customer_id FK", "phone"], accent="#86efac", title_fill="#174331"),
        "orders": draw_table(draw, 290, y + 198, 178, "orders", ["order_id PK", "customer_id FK"], accent="#93c5fd", title_fill="#17324a"),
        "order_items": draw_table(draw, 512, y + 252, 196, "order_items", ["order_id FK", "product_id FK", "quantity"], accent="#c4b5fd", title_fill="#392456"),
        "products": draw_table(draw, 780, y + 178, 190, "products", ["product_id PK", "product_name", "supplier_id FK"], accent="#facc15", title_fill="#4a3a13"),
        "suppliers": draw_table(draw, 780, y + 54, 190, "suppliers", ["supplier_id PK", "supplier"], accent="#facc15", title_fill="#4a3a13"),
    }
    connect(draw, (468, y + 124), (540, y + 124), "#86efac")
    connect(draw, (379, y + 152), (379, y + 198), "#93c5fd")
    connect(draw, (468, y + 274), (512, y + 294), "#c4b5fd")
    connect(draw, (708, y + 294), (780, y + 248), "#facc15")
    connect(draw, (875, y + 178), (875, y + 160), "#facc15")
    section_line(draw, 940)


def draw_ticket_section(draw: ImageDraw.ImageDraw, data: dict[str, object]) -> None:
    y = 966
    draw.text((36, y), str(data["ticket_title"]), font=font(34, bold=True), fill="#93c5fd")

    users = draw_table(draw, 36, y + 62, 140, "users", ["user_id PK", "name", "role"], accent="#93c5fd", title_fill="#16324a", row_h=31, title_h=38)
    tickets = draw_table(draw, 220, y + 52, 170, "tickets", ["ticket_id PK", "customer_id FK", "assignee_id FK", "status", "category_id FK"], accent="#93c5fd", title_fill="#16324a", row_h=31, title_h=38)
    messages = draw_table(draw, 435, y + 62, 180, "ticket_messages", ["message_id PK", "ticket_id FK", "sender_id FK", "message"], accent="#93c5fd", title_fill="#16324a", row_h=31, title_h=38)
    categories = draw_table(draw, 664, y + 52, 160, "categories", ["category_id PK", "name"], accent="#a78bfa", title_fill="#34225a", row_h=31, title_h=38)
    tags = draw_table(draw, 664, y + 206, 132, "tags", ["tag_id PK", "name"], accent="#facc15", title_fill="#433813", row_h=31, title_h=38)
    ticket_tags = draw_table(draw, 430, y + 222, 168, "ticket_tags", ["ticket_id FK", "tag_id FK"], accent="#a78bfa", title_fill="#34225a", row_h=31, title_h=38)
    connect(draw, (176, y + 104), (220, y + 104), "#93c5fd")
    connect(draw, (390, y + 116), (435, y + 116), "#93c5fd")
    connect_path(draw, [(390, y + 210), (408, y + 210), (408, y + 338), (640, y + 338), (640, y + 96), (664, y + 96)], "#a78bfa")
    connect(draw, (390, y + 246), (430, y + 264), "#a78bfa")
    connect(draw, (598, y + 264), (664, y + 248), "#facc15")

    ix_x = 836
    index_title_size = 25 if data.get("locale") == "en" else 28
    title_bottom = draw_text(
        draw,
        (ix_x, y + 2),
        str(data["index_title"]),
        font(index_title_size, bold=True),
        fill="#fde68a",
        max_width=155,
        line_gap=0,
    )
    note_bottom = draw_text(
        draw,
        (ix_x, title_bottom + 6),
        str(data["index_note"]),
        font(16 if data.get("locale") == "en" else 17),
        fill="#fef3c7",
        max_width=150,
        line_gap=1,
    )
    list_y = max(y + 100, note_bottom + 12)
    for index, item in enumerate(data["indexes"]):  # type: ignore[arg-type]
        yy = list_y + index * 46
        rounded(draw, (ix_x, yy, 990, yy + 40), "#292218", "#facc15", width=2, radius=8)
        draw.text((ix_x + 10, yy + 10), "→", font=font(17, bold=True), fill="#facc15")
        draw_text(draw, (ix_x + 34, yy + 7), str(item), font(12, bold=True), fill="#fefce8", max_width=112, line_gap=0)
    section_line(draw, 1362)


def draw_check(draw: ImageDraw.ImageDraw, data: dict[str, object]) -> None:
    y = 1390
    x = badge(draw, 36, y, str(data["check"]), "#7e22ce", font(28, bold=True))
    draw_text(draw, (x, y + 5), f"{data['before_rows'][0]} / {data['after_rows'][0]}", font(23, bold=True), fill="#f3e8ff", max_width=650)  # type: ignore[index]

    left = (52, y + 76, 486, y + 268)
    right = (540, y + 76, 974, y + 268)
    rounded(draw, left, "#1f1416", "#ef4444", width=3, radius=8)
    rounded(draw, right, "#122217", "#22c55e", width=3, radius=8)
    draw.rectangle((left[0], left[1], left[2], left[1] + 42), fill="#3f161b")
    draw.rectangle((right[0], right[1], right[2], right[1] + 42), fill="#17381f")
    draw.text((left[0] + 18, left[1] + 8), str(data["before"]), font=font(23, bold=True), fill="#fecaca")
    draw.text((right[0] + 18, right[1] + 8), str(data["after"]), font=font(23, bold=True), fill="#bbf7d0")
    row_font = font(23, bold=True)
    for i, row in enumerate(data["before_rows"]):  # type: ignore[arg-type]
        yy = left[1] + 58 + i * 44
        draw_cross(draw, left[0] + 22, yy + 5)
        draw_text(draw, (left[0] + 62, yy + 2), str(row), row_font, fill="#fee2e2", max_width=330)
    for i, row in enumerate(data["after_rows"]):  # type: ignore[arg-type]
        yy = right[1] + 58 + i * 44
        draw_tick(draw, right[0] + 22, yy + 3)
        draw_text(draw, (right[0] + 62, yy + 2), str(row), row_font, fill="#dcfce7", max_width=330)
    draw.text((500, y + 146), "→", font=font(42, bold=True), fill="#f8fafc")


def render(locale: str, output_dir: Path) -> Path:
    data = LOCALES[locale]
    image = Image.new("RGB", (W, H), "#0f172a")
    draw = ImageDraw.Draw(image)
    background(draw)
    title_size = {"zh": 54, "en": 42, "ja": 48}[locale]
    subtitle_y = draw_text(draw, (40, 34), str(data["title"]), font(title_size, bold=True), fill="#f8fafc", max_width=940, line_gap=2)
    draw_text(draw, (40, max(104, subtitle_y + 4)), str(data["subtitle"]), font(28, bold=True), fill="#fde68a", max_width=930)
    section_line(draw, 164)
    draw_wide_table(draw, data)
    draw_split_tables(draw, data)
    draw_ticket_section(draw, data)
    draw_check(draw, data)
    draw_text(draw, (40, 1710), str(data["footer"]), font(34, bold=True), fill="#fde68a", max_width=940)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / str(data["file"])
    image.save(out, format="PNG")
    out.chmod(0o644)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="tmp/content-image-sync-20260521-final")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    for locale in ["zh", "en", "ja"]:
        print(render(locale, output_dir))


if __name__ == "__main__":
    main()
