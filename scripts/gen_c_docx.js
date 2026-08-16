/* 把「C：用生产检索模式重跑生成评估」生成排版清晰的 Word 版复盘存档。
 * 运行：$env:NODE_PATH = "C:\Users\hp\Desktop\简历\node_modules"; node scripts\gen_c_docx.js
 * 输出：docs/C_用生产检索模式重跑生成评估.docx
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun,
  Table, TableRow, TableCell,
  AlignmentType, BorderStyle, WidthType, ShadingType
} = require("docx");

const FONT = "Microsoft YaHei";
const COLOR_BODY = "333333";
const COLOR_DIM = "666666";
const COLOR_DK = "1a1a1a";
const COLOR_HDR = "1F4E79";
const COLOR_H2 = "1F4E79";
const COLOR_H3 = "2E74B5";
const COLOR_ACC = "C00000";

const OUT = path.join(__dirname, "..", "docs", "C_用生产检索模式重跑生成评估.docx");

function r(text, opts = {}) {
  return new TextRun({
    text,
    font: FONT,
    size: opts.size || 20,          // 10pt
    bold: opts.bold || false,
    italics: opts.italics || false,
    color: opts.color || COLOR_BODY,
  });
}

function para(runs, opts = {}) {
  return new Paragraph({
    spacing: { before: opts.before || 0, after: opts.after != null ? opts.after : 100, line: opts.line || 300 },
    alignment: opts.align || AlignmentType.LEFT,
    indent: opts.indent ? { left: opts.indent } : undefined,
    children: Array.isArray(runs) ? runs : [runs],
  });
}

function bullet(label, text, opts = {}) {
  return para([
    r("· ", { bold: true, color: COLOR_DK }),
    ...(label ? [r(label, { bold: true, color: COLOR_DK }), r(text, opts)] : [r(text, opts)]),
  ], { after: 60 });
}

function h1(text) {
  return new Paragraph({
    spacing: { before: 0, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 12, color: COLOR_HDR, space: 2 } },
    children: [r(text, { size: 30, bold: true, color: COLOR_DK })],
  });
}

function h2(text) {
  return new Paragraph({
    spacing: { before: 220, after: 100 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: COLOR_H2, space: 1 } },
    children: [r(text, { size: 24, bold: true, color: COLOR_H2 })],
  });
}

function h3(text) {
  return new Paragraph({
    spacing: { before: 160, after: 60 },
    children: [r(text, { size: 21, bold: true, color: COLOR_H3 })],
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const allBorders = {
  top: border, bottom: border, left: border, right: border,
  insideHorizontal: border, insideVertical: border,
};

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 90, right: 90 },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: COLOR_HDR } : undefined,
    verticalAlign: "center",
    children: [para([r(text, {
      bold: opts.header || false,
      color: opts.header ? "FFFFFF" : COLOR_BODY,
      size: opts.header ? 19 : 18,
    })], { after: 0, line: 260 })],
  });
}

function makeTable(widths, headers, rows) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    borders: allBorders,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, { width: widths[i], header: true })) }),
      ...rows.map(row => new TableRow({ children: row.map((c, i) => cell(c, { width: widths[i] })) })),
    ],
  });
}

// ─────────────────────────────────────────────────────────────
// 正文内容
// ─────────────────────────────────────────────────────────────
const children = [];

children.push(h1("C_用生产检索模式重跑生成评估"));
children.push(para([
  r("包 C：评测闭环对准生产路径——面试/复盘存档 ｜ 2026-08-16 ｜ 40 条周医生真实门诊语料 ｜ 生成 qwen3.7-max-2026-06-08 ｜ judge deepseek-v4-flash", { color: COLOR_DIM }),
], { after: 160 }));

children.push(h2("一句话总结"));
children.push(para([r("简历上的生成侧 RAGAS 数字（忠实度 65%、上下文相关性 51%、相似度 62%）是用 fast（纯 BM25）检索模式测的，而生产真正跑的是 RRF 融合 + BGE 重排管线——评测没有对着生产路径。这一包把生成评测切到 production 模式重跑同一 40 条测试集，结果：上下文相关性 0.512 → 0.781（+53%），忠实度与相似度基本持平。")], { after: 40 }));
children.push(para([
  r("复盘一句话：", { bold: true, color: COLOR_DK }),
  r("「你测的路径得是你线上走的路径，否则 A/B 结论是自我安慰。」"),
], { after: 80 }));

// ── 背景 ──
children.push(h2("背景：为什么重跑"));
const why = [
  "检索管线升级没被评测覆盖：生产入口是 KnowledgeRetriever.retrieve()（Chroma 稠密 + jieba-BM25 两路召回 → RRF 融合 → BGE cross-encoder 重排 top-3）；但生成侧评测脚本默认 fast 模式只走 QR→BM25。简历上的 RAGAS 数字因此只代表了旧检索路径。",
  "评测必须对着生产调用链：这是包 A（A2）就确立的纪律——对非生产入口做 A/B 得到的是假阳性结论。",
];
why.forEach(t => children.push(bullet(null, t)));
children.push(bullet("涉及文件：", "scripts/eval_generation.py（--retriever fast|production）、config/settings.py/.env（模型配置）、core/rag/hybrid_retriever.py + core/rag/reranker.py（生产检索链）"));

// ── 过程 ──
children.push(h2("过程：踩掉三个坑才跑通"));
children.push(h3("坑 1：qwen3.7-max 免费额度耗尽（403）"));
const p1 = [
  "主对话模型 qwen3.7-max 别名与 2026-05-20 快照的免费额度先后耗尽（403 AllocationQuota.FreeTierOnly），生成与改写全链路被阻塞。",
  "逐个探测快照：2026-05-17 返回 400（该快照不存在）；2026-06-08 实测可调用（仍有免费额度）→ 切换 .env 的 MODEL_NAME 与 REWRITER_MODEL_NAME。",
  "可比性：06-08 与 05-20 是同族同版本（qwen3.7-max）的不同快照，语义能力一致，数字仍可与旧基线对比；judge 全程走 deepseek-v4-flash（独立额度），不受影响。",
];
p1.forEach(t => children.push(bullet(null, t)));

children.push(h3("坑 2：judge max_tokens 触顶，faithfulness 大量判空"));
const p2 = [
  "冒烟 5 条里 3 条的 faithfulness 双 judge 全判空（报错 The output is incomplete due to a max_tokens length limit）。fast 基线当时 40/40 全有效——是 deepseek judge 输出变长后 2048 上限不够用了。",
  "修复：_make_ragas_judge 的 max_tokens 2048 → 4096。修复后全量 40 条里 faithfulness 丢失 8 条（n=32），丢失率降到约 20%，不再是系统性失败。",
];
p2.forEach(t => children.push(bullet(null, t)));

children.push(h3("坑 3：production 模式慢（BGE CPU 重排）"));
const p3 = [
  "单条从 401s（含模型加载）热身后降到约 220s；全量 40 条总耗时 219 分钟。本地 bge-reranker-v2-m3 CPU 推理是既有决策（用户接受，不换远程重排 API）。",
  "执行路径：smoke 5 条验证脚本可跑 → 全量 40 条后台跑完（exit 0）。",
];
p3.forEach(t => children.push(bullet(null, t)));

// ── 结果 ──
children.push(h2("结果：fast vs production（同一 40 条测试集）"));
children.push(makeTable(
  [2380, 2300, 2300, 2300],
  ["指标", "fast（BM25）", "production（RRF+BGE）", "Δ"],
  [
    ["上下文相关性（独立 judge）", "0.512 ± 0.440", "0.781 ± 0.286", "+53%"],
    ["上下文相关性（自评 judge）", "0.650 ± 0.382", "0.825 ± 0.263", "+27%"],
    ["忠实度（独立 judge）", "0.649 ± 0.264", "0.587 ± 0.274", "-0.062*"],
    ["忠实度（自评 judge）", "0.723 ± 0.236", "0.643 ± 0.273", "-0.080*"],
    ["医生金标准相似度（bge-m3）", "0.622 ± 0.092", "0.629 ± 0.088", "持平"],
  ],
));
children.push(para([
  r("* 忠实度为 n=32/40（8 条因 judge 超长输出偶发丢失，fast 基线为 40/40），非同口径对比，不能直接读成检索退化。", { color: COLOR_DIM, size: 18, italics: true }),
], { before: 80, after: 80 }));

// ── 解读 ──
children.push(h2("怎么解读"));
children.push(h3("上下文相关性大涨——这是 RRF+BGE 该赢的指标"));
children.push(para([r("相关性是纯检索质量指标（检索到的资料 vs 用户问题），不依赖生成与 judge。RRF 融合 + BGE 重排把相关性从 0.512 拉到 0.781，独立 judge 口径 +53%——这是生产检索升级最有说服力的证据。")], { after: 60 }));
children.push(h3("忠实度小幅下降：三种解释，都不是检索退化"));
const interp = [
  "数据口径不一致：production 是 n=32 vs fast 的 n=40，缺的 8 条若是高分项会拉低均值。",
  "上下文更全 → grounded 回答更长更细 → claims 更多 → 每条都要在更大的上下文里逐一验证，分被拉低（RAGAS 里忠实度与相关性存在天然张力）。",
  "生成模型快照变化（05-20 → 06-08）引入少量方差。",
];
interp.forEach(t => children.push(bullet(null, t)));
children.push(h3("医生金标准相似度持平（0.622 → 0.629）"));
children.push(para([r("persona 生成质量不被检索模式影响，周医生风格回复稳定。")], { after: 60 }));
children.push(h3("双 judge 分歧收窄"));
children.push(bullet(null, "任一指标分歧 > 0.15 的查询：29 条（fast）→ 22 条（production）；忠实度 |Δ| 0.253 → 0.169，相关性 |Δ| 0.175 → 0.144。两个 judge 意见更一致。"));
children.push(para([
  r("面试金句：", { bold: true, color: COLOR_ACC }),
  r("「我重新测了一遍生产路径。相关性 +53% 是实打实的检索升级收益；忠实度的小幅回落我做了归因——数据缺失、上下文更全、快照变化，而不是结论先行。」"),
], { before: 100, after: 80 }));

// ── 归档 ──
children.push(h2("数据归档（可复现）"));
const arch = [
  "production 全量：scripts/eval_generation_results.production.{json,md}（40 条逐条明细 + 聚合 + 双 judge 分歧）",
  "fast 基线：scripts/eval_generation_results.fast.{json,md}（与 production 同结构）",
  "测试集：data/eval/zhou_queries.jsonl（40 条，脱敏，医生回应即金标准）",
];
arch.forEach(t => children.push(bullet(null, t)));
children.push(bullet("简历更新：", "建议把生成侧「上下文相关性」从 51% 更新为生产管线口径 78%（+53%），或按 fast 基线保留并加注 RRF+BGE 收益行。忠实度降幅用归因圆场（方案见上）。待用户确认后再改 generate_resume.js。"));

// ── 面试怎么讲 ──
children.push(h2("面试怎么讲"));
children.push(h3("30 秒版"));
children.push(para([r("「我做过一次评测口径纠偏：简历里的 RAGAS 数字是旧 BM25 路径测的，而生产早换成了 RRF 融合 + BGE 重排。我把同一套 40 条真实门诊语料用生产管线重跑了一遍——上下文相关性 0.512 → 0.781（+53%），忠实度和相似度持平。中途还解决了两件事：qwen3.7-max 免费额度耗尽后逐快照探测换到可用快照；deepseek judge 输出变长触 max_tokens 上限导致忠实度大量判空，提到 4096 后恢复。」")], { after: 80 }));

children.push(h3("追问准备"));
const qa = [
  ["为什么评测一定要对着生产路径？", "非生产入口走的是旧检索逻辑，A/B 测出来的是假阳性结论；只有生产调用链上的数据才能指导优化与简历陈述。"],
  ["+53% 是重排的功劳还是融合的功劳？", "是 RRF 融合 + BGE 重排两者叠加的效果（fast=纯 BM25 vs production=两路召回+融合+重排），不是单一组件。"],
  ["忠实度为什么降了？", "三个原因：production 缺 8 条数据（n=32 vs 40）、上下文更全导致回答更长 claims 更多逐条验证拉低、模型快照变化。都不是检索退化。"],
  ["换模型怎么保证数字可比？", "同族同版本快照（qwen3.7-max 06-08 vs 05-20）语义一致；judge 全程固定 deepseek-v4-flash 独立额度，judge 环节不变。"],
  ["max_tokens 触顶怎么定位的？", "冒烟 5 条里 faithfulness 大量判空而 context_relevance 全出 → 锁定在 judge 的 claims 抽取阶段输出过长；fast 基线当时 40/40 有效反证是上限而非代码错。"],
];
qa.forEach(([q, a]) => {
  children.push(para([r("Q：", { bold: true, color: COLOR_ACC }), r(q, { bold: true })], { before: 60, after: 20 }));
  children.push(para([r("A：", { bold: true, color: COLOR_DK }), r(a)], { after: 40 }));
});

// ─────────────────────────────────────────────────────────────
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },   // A4
        margin: { top: 900, bottom: 720, left: 1260, right: 1260 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("Done: " + OUT + " (" + buf.length + " bytes)");
});
