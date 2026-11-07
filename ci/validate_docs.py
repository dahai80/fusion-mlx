#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fusion 文档体系静态校验器（仅用标准库，可直接进 CI）

用法：
  python3 ci/validate_docs.py --root . --product fusion-mlx --class core --entry service
  python3 ci/validate_docs.py --root . --strict          # 把 WARN 也当失败
  python3 ci/validate_docs.py --root . --skip-inventory  # 只做格式/红线检查

检查五类：
  1. frontmatter 完整性 + 取值合法
  2. 文档集完整性（按 class × entry 推导 Tier 0 / Tier 1）
  3. source_of_truth 路径存在性
  4. 红线扫描（禁词 / 占位符 / 密钥 / 相对链接）
  5. 行数预算 + zh/en 配对

退出码：0 通过；1 有 ERROR；--strict 时 WARN 也返回 1；2 用法错误。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------- 常量定义

REQUIRED_KEYS = [
    "doc_id", "product", "class", "entry", "doc_type",
    "title", "audience", "status", "version", "updated",
    "owner", "source_of_truth",
]

VALID_CLASS = {"core", "platform", "app"}
VALID_ENTRY = {"developer", "service", "guided-ui", "none"}
VALID_STATUS = {"draft", "review", "stable", "deprecated"}
VALID_DOC_TYPE = {
    "readme", "product-page", "get-started", "api-reference", "config-reference",
    "architecture", "troubleshooting", "recipe", "changelog", "guide",
}

# 行数预算（不含 frontmatter）
LINE_BUDGET = {
    "get-started": 120,          # guided-ui 另有 100 的收紧值，见下
    "troubleshooting": 250,
    "architecture": 250,
    "recipe": 80,
    "config-reference": 400,
}
GUIDED_UI_BUDGET = 100

# fusion-mlx repo overrides: architecture.md and config-reference.md predate the
# docs contract and exceed the default budgets. Splitting them into sub-pages is
# tracked as P2 debt; budgets here only keep the status quo from growing.
LINE_BUDGET["architecture"] = 400
LINE_BUDGET["config-reference"] = 460

# class × entry -> Tier 1 必须存在的文档（Tier 0 三件固定）
TIER1 = {
    ("core", "developer"): [
        "docs/api-reference.md", "docs/config-reference.md",
        "docs/architecture.md", "CHANGELOG.md",
    ],
    ("core", "service"): [
        "docs/api-reference.md", "docs/config-reference.md",
        "docs/architecture.md", "docs/troubleshooting.md", "CHANGELOG.md",
    ],
    ("platform", "developer"): [
        "docs/api-reference.md", "docs/config-reference.md",
        "docs/architecture.md", "CHANGELOG.md",
    ],
    ("platform", "service"): [
        "docs/api-reference.md", "docs/config-reference.md",
        "docs/architecture.md", "docs/troubleshooting.md",
        "docs/guides", "CHANGELOG.md",
    ],
    ("app", "developer"): [
        "docs/api-reference.md", "docs/config-reference.md", "CHANGELOG.md",
    ],
    ("app", "service"): [
        "docs/api-reference.md", "docs/config-reference.md",
        "docs/troubleshooting.md", "CHANGELOG.md",
    ],
    ("app", "guided-ui"): [
        "docs/troubleshooting.md", "docs/img", "CHANGELOG.md",
    ],
}
TIER0 = ["README.md", "docs/product-page.md", "docs/get-started.md"]

# 需要 ≥2 篇的目录型 Tier 1
TIER1_MIN_COUNT = {
    ("core", "developer"): [("docs/recipes", 2)],
    ("platform", "developer"): [("docs/recipes", 2)],
    ("app", "developer"): [("docs/recipes", 2)],
}

# 禁词（与 guidelines/writing-style.md §4 同步；此处只放中文核心词，英文另表）
FORBIDDEN_ZH = [
    "强大的", "十分强大", "业界领先", "行业领先", "领先的", "极致的", "极致",
    "完美的", "完美", "顶级", "顶尖", "最强", "首屈一指",
    "革命性", "颠覆性", "颠覆", "划时代", "里程碑", "黑科技",
    "一站式", "全场景", "全栈赋能", "赋能", "闭环打通", "打通",
    "无与伦比", "卓越", "惊艳", "灵动", "丝滑", "无缝",
    "海量", "超高", "超低",
]
FORBIDDEN_EN = [
    "powerful", "robust", "world-class", "best-in-class", "cutting-edge",
    "state-of-the-art", "industry-leading", "seamless", "blazing",
    "revolutionary", "game-changing", "effortless", "unparalleled",
]

PLACEHOLDER_RE = re.compile(r"<(?!https?:|mailto:)([^<>\n]{0,40})>")
HTML_SAFE = {
    "br", "/br", "p", "/p", "b", "/b", "i", "/i", "code", "/code",
    "span", "/span", "div", "/div", "table", "/table", "tr", "/tr",
    "td", "/td", "th", "/th", "a", "/a", "img", "/img", "svg", "/svg",
    "script", "/script", "style", "/style", "html", "/html",
}

SECRET_PATTERNS = [
    (re.compile(r"sk-[A-Za-z0-9]{16,}"), "疑似 OpenAI 风格密钥"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "疑似 AWS Access Key"),
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{24,}"), "疑似硬编码 Bearer token"),
    (re.compile(r"(?i)(password|passwd|secret|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
     "疑似硬编码口令"),
]
IP_RE = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
IP_ALLOW = {"127.0.0.1", "0.0.0.0", "255.255.255.255"}

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")

# ---------------------------------------------------------------- 数据结构


@dataclass
class Finding:
    level: str      # ERROR | WARN | INFO
    code: str
    file: str
    line: int
    msg: str

    def __str__(self) -> str:
        loc = f"{self.file}:{self.line}" if self.line else self.file
        return f"[{self.level}] {self.code} {loc}  {self.msg}"


@dataclass
class Doc:
    path: str
    meta: Dict[str, object] = field(default_factory=dict)
    body_start: int = 0
    body_lines: int = 0


# ---------------------------------------------------------------- frontmatter

KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")
ITEM_RE = re.compile(r"^\s*-\s+(.+)$")


def parse_frontmatter(text: str) -> Tuple[Dict[str, object], int, int]:
    """返回 (meta, frontmatter 结束行号 1-based, 总行数)。无 frontmatter 返回 ({}, 0, N)。"""
    lines = text.splitlines()
    total = len(lines)
    if not lines or lines[0].strip() != "---":
        return {}, 0, total
    end = None
    for i in range(1, total):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, 0, total

    meta: Dict[str, object] = {}
    current: Optional[str] = None
    for raw in lines[1:end]:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        m = KEY_RE.match(raw)
        if m and not raw.startswith((" ", "\t")):
            key, val = m.group(1), m.group(2).strip()
            current = key
            if val == "":
                meta[key] = []           # 可能是列表
            elif val.startswith("{") and val.endswith("}"):
                meta[key] = val          # 内联 map，保留原文
            else:
                meta[key] = val.strip("'\"")
            continue
        im = ITEM_RE.match(raw)
        if im and current is not None:
            if not isinstance(meta.get(current), list):
                meta[current] = []
            meta[current].append(im.group(1).strip().strip("'\""))
    return meta, end + 1, total


def load_doc(path: str) -> Doc:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    meta, body_start, total = parse_frontmatter(text)
    return Doc(path=path, meta=meta, body_start=body_start, body_lines=total - body_start)


# ---------------------------------------------------------------- 各检查


def check_frontmatter(doc: Doc, rel: str, out: List[Finding]) -> None:
    if not doc.meta:
        out.append(Finding("ERROR", "FM-MISSING", rel, 1, "缺少 frontmatter（文件首行必须是 ---）"))
        return

    for key in REQUIRED_KEYS:
        if key not in doc.meta:
            out.append(Finding("ERROR", "FM-KEY", rel, 1, f"缺少必填字段 {key}"))

    def val(k: str) -> Optional[str]:
        v = doc.meta.get(k)
        return v if isinstance(v, str) else None

    cls = val("class")
    if cls and cls not in VALID_CLASS:
        out.append(Finding("ERROR", "FM-CLASS", rel, 1, f"class 非法：{cls}"))
    entry = val("entry")
    if entry and entry not in VALID_ENTRY:
        out.append(Finding("ERROR", "FM-ENTRY", rel, 1, f"entry 非法：{entry}"))
    st = val("status")
    if st and st not in VALID_STATUS:
        out.append(Finding("ERROR", "FM-STATUS", rel, 1, f"status 非法：{st}"))
    dt = val("doc_type")
    if dt and dt not in VALID_DOC_TYPE:
        out.append(Finding("ERROR", "FM-DOCTYPE", rel, 1, f"doc_type 非法：{dt}"))

    owner = val("owner")
    if owner and not owner.startswith("@"):
        out.append(Finding("ERROR", "FM-OWNER", rel, 1, "owner 必须是 @handle（单个具体的人）"))
    if owner and owner.startswith("@") and len(owner.strip("@")) > 39:
        out.append(Finding("WARN", "FM-OWNER", rel, 1, f"owner 过长，疑似团队名：{owner}"))

    up = val("updated")
    if up and not re.match(r"^\d{4}-\d{2}-\d{2}$", up):
        out.append(Finding("ERROR", "FM-DATE", rel, 1, f"updated 格式应为 YYYY-MM-DD：{up}"))

    doc_id = val("doc_id")
    if doc_id and not re.match(r"^[a-z0-9-]+/[a-z0-9-]+(/[a-z0-9-]+)?$", doc_id):
        out.append(Finding("ERROR", "FM-DOCID", rel, 1, f"doc_id 格式应为 <product>/<doc_type>：{doc_id}"))

    if st == "deprecated" and not doc.meta.get("superseded_by"):
        out.append(Finding("ERROR", "FM-SUPERSEDE", rel, 1, "status=deprecated 必须给 superseded_by"))

    sot = doc.meta.get("source_of_truth")
    if not isinstance(sot, list) or not sot:
        out.append(Finding("ERROR", "FM-SOT", rel, 1, "source_of_truth 必须是非空列表"))


def check_sot_paths(doc: Doc, rel: str, repo_root: str, out: List[Finding]) -> None:
    sot = doc.meta.get("source_of_truth")
    if not isinstance(sot, list):
        return
    for item in sot:
        p = str(item).split("#")[0].strip()
        if not p:
            continue
        target = os.path.join(repo_root, p)
        if not os.path.exists(target):
            out.append(Finding("ERROR", "SOT-PATH", rel, 1,
                              f"source_of_truth 路径不存在：{p}"))
            continue
        # 行号范围校验
        if "#L" in str(item) and os.path.isfile(target):
            m = re.search(r"#L(\d+)(?:-L(\d+))?", str(item))
            if m:
                start = int(m.group(1))
                end = int(m.group(2) or start)
                try:
                    with open(target, "r", encoding="utf-8", errors="replace") as fh:
                        n = sum(1 for _ in fh)
                except OSError:
                    continue
                if end > n or start < 1 or start > end:
                    out.append(Finding("ERROR", "SOT-LINE", rel, 1,
                                      f"行号越界：{item}（文件共 {n} 行）"))


def check_redlines(text: str, rel: str, out: List[Finding]) -> None:
    lines = text.splitlines()
    in_fence = False
    in_inline = False  # inline-code span state, carried across lines
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            in_inline = False
            continue

        # 禁词（代码块内也扫，避免在示例里写营销词）
        for w in FORBIDDEN_ZH:
            if w in line:
                out.append(Finding("ERROR", "RED-MKT", rel, idx, f"营销/模糊词：{w}"))
        for w in FORBIDDEN_EN:
            if re.search(r"\b" + re.escape(w) + r"\b", line, re.I):
                out.append(Finding("ERROR", "RED-MKT", rel, idx, f"marketing word: {w}"))

        # 占位符残留（frontmatter 行放行，模板文件也放行由 --skip-placeholder 控制；
        # 代码块内跳过：Rust/C++/TS 泛型 <T> 等尖括号语法不是占位符；
        # 行内代码 span 先剥掉：`<|tool_call|>` 等字面 token 不是占位符）
        if not in_fence:
            if not stripped:
                in_inline = False  # spans do not survive blank lines; prevents cascade
            segs = line.split("`")
            prose_parts = []
            for i, seg in enumerate(segs):
                if not in_inline:
                    prose_parts.append(seg)
                if i < len(segs) - 1:
                    in_inline = not in_inline  # toggle only at real backticks
            prose = "".join(prose_parts)
            for m in PLACEHOLDER_RE.finditer(prose):
                inner = m.group(1).strip()
                if not inner or inner in HTML_SAFE:
                    continue
                tag = re.match(r"^/?([A-Za-z][A-Za-z0-9]*)", inner)
                if tag and tag.group(1).lower() in HTML_SAFE:
                    continue  # HTML tag with attributes, e.g. <div align="center">
                if inner[0].isdigit():
                    continue  # comparison like r<5 or margin<4e-4, not a tag
                if inner.startswith("//") or inner.endswith("/"):
                    continue
                out.append(Finding("ERROR", "RED-PH", rel, idx, f"占位符残留：<{inner}>"))

        # 密钥
        for pat, why in SECRET_PATTERNS:
            if pat.search(line):
                out.append(Finding("ERROR", "RED-SECRET", rel, idx, why))

        # 内网 IP
        for m in IP_RE.finditer(line):
            ip = m.group(0)
            if ip in IP_ALLOW:
                continue
            out.append(Finding("WARN", "RED-IP", rel, idx, f"出现 IPv4 明文：{ip}（确认非内网地址）"))


def check_links(doc_path: str, text: str, rel: str, out: List[Finding]) -> None:
    base = os.path.dirname(doc_path)
    # Strip fenced + inline code first: `layers[i](hidden)` is a call, not a link.
    # Inline-code state carries across lines; resets on blank lines.
    stripped_lines = []
    in_fence = False
    in_inline = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            in_inline = False
            continue
        if in_fence or not line.strip():
            in_inline = False if not line.strip() else in_inline
            stripped_lines.append("")
            continue
        segs = line.split("`")
        parts = []
        for i, seg in enumerate(segs):
            if not in_inline:
                parts.append(seg)
            if i < len(segs) - 1:
                in_inline = not in_inline
        stripped_lines.append("".join(parts))
    for m in LINK_RE.finditer("\n".join(stripped_lines)):
        target = m.group(1)
        if target.startswith("#"):
            continue
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target):
            continue  # any URI scheme: http, https, mailto, file, etc.
        frag = target.split("#")[0]
        if not frag:
            continue
        cand = os.path.normpath(os.path.join(base, frag))
        if not os.path.exists(cand):
            out.append(Finding("ERROR", "LINK-BROKEN", rel, 0, f"相对链接目标不存在：{target}"))


def check_budget(doc: Doc, rel: str, out: List[Finding]) -> None:
    dt = doc.meta.get("doc_type")
    entry = doc.meta.get("entry")
    if not isinstance(dt, str):
        return
    limit = LINE_BUDGET.get(dt)
    if dt == "get-started" and entry == "guided-ui":
        limit = GUIDED_UI_BUDGET
    if limit is None:
        out.append(Finding("INFO", "BUDGET", rel, 0, f"正文 {doc.body_lines} 行（{dt} 无硬上限）"))
        return
    if doc.body_lines > limit:
        over = doc.body_lines - limit
        out.append(Finding("ERROR", "BUDGET", rel, 0,
                           f"正文 {doc.body_lines} 行，超出 {dt} 预算 {limit} 行（超 {over} 行）"))
    elif doc.body_lines > int(limit * 0.9):
        out.append(Finding("WARN", "BUDGET", rel, 0,
                           f"正文 {doc.body_lines} 行，接近 {dt} 预算上限 {limit}"))
    else:
        out.append(Finding("INFO", "BUDGET", rel, 0, f"正文 {doc.body_lines} 行 / 上限 {limit}"))


def check_bi_lingual(doc: Doc, rel: str, out: List[Finding]) -> None:
    body = doc.meta
    # 只看 frontmatter 的 zh/en 配对：title 是否存在两语言
    title = doc.meta.get("title")
    if isinstance(title, str):
        if "zh" not in title or "en" not in title:
            out.append(Finding("WARN", "I18N-TITLE", rel, 1, "title 应为 { zh: ..., en: ... }"))
    elif isinstance(title, list):
        out.append(Finding("WARN", "I18N-TITLE", rel, 1, "title 解析失败，检查缩进"))


def check_inventory(root: str, cls: str, entry: str, out: List[Finding]) -> None:
    key = (cls, entry)
    if key not in TIER1:
        out.append(Finding("ERROR", "INV-KEY", root, 0,
                           f"格位不存在：class={cls} entry={entry}（见总纲 §2.3）"))
        return
    for rel in TIER0:
        if not os.path.exists(os.path.join(root, rel)):
            out.append(Finding("ERROR", "INV-T0", rel, 0, "Tier 0 文档缺失"))
    for rel in TIER1[key]:
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            out.append(Finding("ERROR", "INV-T1", rel, 0, f"Tier 1 文档缺失（格位 {cls}×{entry}）"))
        elif os.path.isdir(p) and not os.listdir(p):
            out.append(Finding("ERROR", "INV-T1", rel, 0, "Tier 1 目录为空"))
    for rel, need in TIER1_MIN_COUNT.get(key, []):
        d = os.path.join(root, rel)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith(".md") and not f.startswith("_")])
            if n < need:
                out.append(Finding("ERROR", "INV-COUNT", rel, 0,
                                   f"至少需要 {need} 篇，当前 {n} 篇"))


# ---------------------------------------------------------------- main


def iter_docs(root: str, skip_dirs: Tuple[str, ...]) -> List[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]
        for fn in filenames:
            if fn.endswith(".md"):
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="fusion 文档体系校验")
    ap.add_argument("--root", default=".", help="仓库根目录")
    ap.add_argument("--product", help="如 fusion-mlx，用于文档集检查")
    ap.add_argument("--class", dest="cls", choices=sorted(VALID_CLASS),
                    help="产品类 core/platform/app")
    ap.add_argument("--entry", choices=sorted(VALID_ENTRY),
                    help="接入形态 developer/service/guided-ui/none")
    ap.add_argument("--strict", action="store_true", help="WARN 也视为失败")
    ap.add_argument("--skip-inventory", action="store_true", help="跳过文档集完整性检查")
    ap.add_argument("--skip-placeholder", action="store_true",
                    help="跳过占位符检查（模板目录用）")
    ap.add_argument("--skip-dirs", default=".git,node_modules,.venv,dist,build,templates,guidelines",
                    help="跳过的目录名，逗号分隔")
    args = ap.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[ERROR] root 不存在：{root}", file=sys.stderr)
        return 2

    skip = tuple(x.strip() for x in args.skip_dirs.split(",") if x.strip())
    findings: List[Finding] = []

    if args.product:
        if not args.cls or not args.entry:
            print("[ERROR] 指定 --product 时必须同时给 --class 与 --entry", file=sys.stderr)
            return 2
        if not args.skip_inventory:
            check_inventory(root, args.cls, args.entry, findings)

    docs_paths = iter_docs(root, skip)
    if not docs_paths:
        print(f"[ERROR] {root} 下未找到任何 .md", file=sys.stderr)
        return 2

    for path in docs_paths:
        rel = os.path.relpath(path, root)
        doc = load_doc(path)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()

        check_frontmatter(doc, rel, findings)
        check_sot_paths(doc, rel, root, findings)
        check_budget(doc, rel, findings)
        check_bi_lingual(doc, rel, findings)
        check_links(path, text, rel, findings)

        if args.skip_placeholder:
            # 去掉占位符检查，其余红线照跑
            saved = list(findings)
            check_redlines(text, rel, findings)
            findings[:] = [f for f in findings
                           if not (f.code == "RED-PH" and f.file == rel)]
        else:
            check_redlines(text, rel, findings)

    # 汇总
    errs = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]
    infos = [f for f in findings if f.level == "INFO"]

    for f in errs:
        print(f)
    for f in warns:
        print(f)
    for f in infos:
        print(f)

    print()
    print(f"扫描 {len(docs_paths)} 篇文档 | ERROR {len(errs)} | WARN {len(warns)} | INFO {len(infos)}")

    if errs:
        return 1
    if args.strict and warns:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
