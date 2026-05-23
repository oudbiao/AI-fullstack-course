const textLanguages = new Set(["", "text", "txt", "plain", "plaintext"]);

const evidenceContextMarkers = [
  "evidence card",
  "evidence to keep",
  "proof of learning",
  "keep evidence",
  "留下的证据",
  "证据卡",
  "留证据",
  "残す証拠",
  "証拠カード",
];

const terminalContextMarkers = [
  "expected output",
  "expected output shape",
  "output starts like",
  "output will look",
  "expected output will look",
  "example output",
  "sample output",
  "output example",
  "example result",
  "预期输出",
  "输出大致",
  "输出类似",
  "输出会像",
  "输出示例",
  "示例输出",
  "示例结果",
  "运行结果",
  "执行结果",
  "出力例",
  "サンプル出力",
  "期待される出力",
  "実行結果",
];

const evidenceLabelHints = [
  "artifact",
  "baseline",
  "deliverable",
  "evidence",
  "failure",
  "first_evidence",
  "first_route",
  "improved",
  "input",
  "metric",
  "output",
  "portfolio",
  "project_thread",
  "risk",
  "score",
  "solution",
  "target_role",
  "target role",
  "technical",
  "trace",
  "user problem",
  "基线",
  "产出",
  "成果",
  "风险",
  "交付",
  "评分",
  "失败",
  "失败检查",
  "岗位",
  "机器学习问题",
  "目标",
  "基线",
  "技术",
  "日志",
  "问题",
  "期望产出",
  "输出",
  "证据",
  "追踪",
  "主线",
  "成果",
  "失敗",
  "期待",
  "証拠",
  "職種",
  "プロジェクト軸",
  "メインルート",
  "評価",
  "改善",
  "課題",
  "目標",
];

const strongEvidenceLabelHints = [
  "artifact",
  "deliverable",
  "evidence",
  "first_evidence",
  "first_route",
  "portfolio",
  "project_thread",
  "target_role",
  "target role",
  "user problem",
  "产出",
  "交付",
  "岗位",
  "证据",
  "期望产出",
  "首条主线",
  "首次证据",
  "项目线",
  "目标岗位",
  "期待成果",
  "証拠",
  "職種",
  "成果物",
  "ユーザー課題",
  "目標職種",
  "プロジェクト軸",
  "最初のメインルート",
  "最初の証拠",
  "期待される成果",
];

const evidenceCardCopy = {
  en: {
    ariaLabel: "Learning evidence card",
    placeholder: "To be filled",
  },
  zh: {
    ariaLabel: "学习证据卡",
    placeholder: "待填写",
  },
  ja: {
    ariaLabel: "学習証拠カード",
    placeholder: "未記入",
  },
};

function extractText(node) {
  if (!node) return "";
  if (typeof node.value === "string") return node.value;
  if (!Array.isArray(node.children)) return "";
  return node.children.map(extractText).join(" ");
}

function hasMarker(text, markers) {
  const normalized = text.toLowerCase();
  return markers.some((marker) => normalized.includes(marker.toLowerCase()));
}

function normalizeLabel(label) {
  return label.trim().toLowerCase().replace(/\s+/g, "_");
}

function hasLabelHint(label, hints) {
  const normalized = normalizeLabel(label);
  return hints.some((hint) => normalized.includes(hint.toLowerCase()));
}

function parseEntries(value) {
  const entries = [];
  let currentEntry = null;

  for (const rawLine of value.split(/\r?\n/)) {
    const line = rawLine.trim().replace(/^[-*]\s+/, "");
    if (!line) continue;

    const match = line.match(/^([^:：]{1,48})[:：]\s*(.*)$/);
    if (match) {
      currentEntry = {
        label: match[1].trim(),
        value: match[2].trim(),
      };
      entries.push(currentEntry);
      continue;
    }

    if (!currentEntry) return [];
    currentEntry.value = `${currentEntry.value} ${line}`.trim();
  }

  return entries.filter((entry) => entry.label);
}

function parseFlowSteps(value) {
  const trimmed = value.trim();
  if (!trimmed || trimmed.split(/\r?\n/).length !== 1) return [];
  if (!/(?:->|→)/.test(trimmed)) return [];

  const steps = trimmed
    .split(/\s*(?:->|→)\s*/g)
    .map((part) => part.trim())
    .filter(Boolean);

  if (steps.length < 3) return [];
  if (steps.some((step) => step.length > 72)) return [];
  return steps;
}

function shouldTransformEvidenceCard(node, previousText, entries) {
  if (!textLanguages.has((node.lang ?? "").toLowerCase())) return false;
  if (entries.length < 2) return false;

  if (hasMarker(previousText, evidenceContextMarkers)) return true;

  const labelHintCount = entries.filter((entry) => hasLabelHint(entry.label, evidenceLabelHints)).length;
  const hasStrongEvidenceLabel = entries.some((entry) => hasLabelHint(entry.label, strongEvidenceLabelHints));

  return entries.length >= 4 && hasStrongEvidenceLabel && labelHintCount >= Math.min(entries.length, 4);
}

function previousCodeLooksExecutable(value) {
  if (!value) return false;

  return /(?:^|\n)\s*(?:print|console\.log|logger\.(?:debug|info|warning|error)|logging\.(?:debug|info|warning|error))\s*\(/.test(
    value,
  ) || /(?:^|\n)\s*(?:echo|printf)\b/.test(value);
}

function looksLikeProgramOutput(value) {
  const text = value.trim();
  if (!text) return false;

  const lines = text.split(/\r?\n/).map((line) => line.trimEnd()).filter(Boolean);
  if (!lines.length) return false;

  const sourceLikeLines = lines.filter((line) =>
    /^(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+\(|class\s+\w+|for\s+.+:|while\s+.+:|if\s+.+:|elif\s+.+:|else:|try:|except\b|return\b|with\s+.+:)/.test(
      line.trim(),
    ),
  );
  if (sourceLikeLines.length >= Math.max(2, Math.ceil(lines.length * 0.35))) return false;

  const outputLikeLines = lines.filter((line) => {
    const trimmed = line.trim();
    return (
      /^Traceback \(most recent call last\):/.test(trimmed) ||
      /^(?:\.\.\.|={3,}|-{3,})$/.test(trimmed) ||
      /^[\[{(].*[\]})]$/.test(trimmed) ||
      /^[^:：]{1,56}[:：]\s+\S/.test(trimmed) ||
      /^\S+\s*=\s*\S+/.test(trimmed) ||
      /(?:=>|->)/.test(trimmed) ||
      /^[-+]?\d+(?:\.\d+)?(?:\s+[-+]?\d+(?:\.\d+)?){1,}$/.test(trimmed)
    );
  });

  return outputLikeLines.length > 0 || lines.length <= 2;
}

function shouldTransformTerminal(node, previousText, previousCodeText) {
  if (!textLanguages.has((node.lang ?? "").toLowerCase())) return false;
  if (!node.value?.trim()) return false;

  if (hasMarker(previousText, terminalContextMarkers)) return true;

  return previousCodeLooksExecutable(previousCodeText) && looksLikeProgramOutput(node.value);
}

function shouldTransformFlowLine(node, steps) {
  if (!textLanguages.has((node.lang ?? "").toLowerCase())) return false;
  return steps.length >= 3;
}

function findAdjacentCodeValue(siblings, index) {
  for (let cursor = index - 1; cursor >= 0; cursor -= 1) {
    const candidate = siblings[cursor];
    if (candidate.type === "code") return candidate.value ?? "";

    const text = extractText(candidate).trim();
    if (!text) continue;
    if (hasMarker(text, terminalContextMarkers)) continue;

    return "";
  }

  return "";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function humanizeLabel(label) {
  const trimmed = label.trim();
  if (!/^[A-Za-z0-9_\-\s/]+$/.test(trimmed)) return trimmed;

  return trimmed
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function localeForFile(file) {
  const filePath = file?.path ?? file?.history?.[0] ?? "";
  if (/[/\\]src[/\\]content[/\\]docs[/\\]zh-cn[/\\]/.test(filePath)) return "zh";
  if (/[/\\]src[/\\]content[/\\]docs[/\\]ja[/\\]/.test(filePath)) return "ja";
  return "en";
}

function formatValue(value, copy) {
  if (!value.trim()) {
    return `<span class="course-evidence-card__placeholder">${escapeHtml(copy.placeholder)}</span>`;
  }

  return escapeHtml(value).replaceAll("-&gt;", '<span class="course-evidence-card__arrow">→</span>');
}

function toEvidenceCardHtml(entries, copy) {
  const rows = entries
    .map((entry) => {
      const label = escapeHtml(humanizeLabel(entry.label));
      const value = formatValue(entry.value, copy);

      return [
        '  <div class="course-evidence-card__row">',
        `    <dt>${label}</dt>`,
        `    <dd>${value}</dd>`,
        "  </div>",
      ].join("\n");
    })
    .join("\n");

  return `<dl class="course-evidence-card" aria-label="${escapeHtml(copy.ariaLabel)}">\n${rows}\n</dl>`;
}

function appendMetaOption(meta, option) {
  const current = (meta ?? "").trim();
  return current ? `${current} ${option}` : option;
}

function toTerminalOutputNode(node) {
  return {
    ...node,
    lang: node.lang || "text",
    meta: appendMetaOption(node.meta, 'frame="terminal"'),
    value: (node.value ?? "").trimEnd(),
  };
}

function toFlowLineHtml(steps) {
  const content = steps
    .map((step, index) => {
      const item = `<span class="course-flow-line__step">${escapeHtml(step)}</span>`;
      if (index === steps.length - 1) return item;
      return `${item}<span class="course-flow-line__arrow" aria-hidden="true">→</span>`;
    })
    .join("");

  return `<div class="course-flow-line" aria-label="Process flow">${content}</div>`;
}

function transformCourseTextBlocks(parent, copy) {
  if (!Array.isArray(parent.children)) return;

  for (let index = 0; index < parent.children.length; index += 1) {
    const node = parent.children[index];

    if (node.type === "code") {
      const previousText = parent.children
        .slice(Math.max(0, index - 8), index)
        .map(extractText)
        .join(" ");
      const immediatePreviousText = extractText(parent.children[index - 1] ?? "");
      const previousCodeText = findAdjacentCodeValue(parent.children, index);
      const entries = parseEntries(node.value ?? "");
      const flowSteps = parseFlowSteps(node.value ?? "");

      if (shouldTransformEvidenceCard(node, previousText, entries)) {
        parent.children[index] = {
          type: "html",
          value: toEvidenceCardHtml(entries, copy),
        };
        continue;
      }

      if (shouldTransformTerminal(node, immediatePreviousText, previousCodeText)) {
        parent.children[index] = toTerminalOutputNode(node);
        continue;
      }

      if (shouldTransformFlowLine(node, flowSteps)) {
        parent.children[index] = {
          type: "html",
          value: toFlowLineHtml(flowSteps),
        };
      }

      continue;
    }

    transformCourseTextBlocks(node, copy);
  }
}

export default function remarkCourseTextBlocks() {
  return (tree, file) => {
    const copy = evidenceCardCopy[localeForFile(file)] ?? evidenceCardCopy.en;
    transformCourseTextBlocks(tree, copy);
  };
}
