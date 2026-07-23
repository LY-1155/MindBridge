"""从 dayi.org.cn 抓取药品说明书数据，用于 medication_knowledge.jsonl 的双源交叉验证。

用法：
  python scripts/scrape_dayi_medication.py

输出：scripts/dayi_scraped_data.json (结构化提取)
      scripts/dayi_scraped_raw/ (原始HTML备份)
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_JSON = PROJECT_ROOT / "scripts" / "dayi_scraped_data.json"
RAW_DIR = PROJECT_ROOT / "scripts" / "dayi_scraped_raw"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}

# ---------------------------------------------------------------------------
# 要抓取的药物列表（每药一个主页面）
# ---------------------------------------------------------------------------
DRUGS = {
    "西酞普兰": {
        "en": "citalopram",
        "urls": [
            "https://m.dayi.org.cn/drug/1156779",
            "http://s.dayi.org.cn/mip/s/medical/1152039.html",
        ],
    },
    "氟伏沙明": {
        "en": "fluvoxamine",
        "urls": [
            "https://m.dayi.org.cn/drug/1152221",
            "http://s.dayi.org.cn/mip/s/medical/1152221.html",
        ],
    },
    "安非他酮": {
        "en": "bupropion",
        "urls": [
            "https://m.dayi.org.cn/drug/1152175",
            "http://s.dayi.org.cn/mip/s/medical/1152175.html",
        ],
    },
    "曲唑酮": {
        "en": "trazodone",
        "urls": [
            "https://m.dayi.org.cn/drug/1152220",
            "http://s.dayi.org.cn/mip/s/medical/1152220.html",
        ],
    },
    # ---------- 第三批：非典型抗精神病药 ----------
    "奥氮平": {
        "en": "olanzapine",
        "urls": [
            "https://m.dayi.org.cn/drug/1152139",
            "http://s.dayi.org.cn/mip/s/medical/1152139.html",
        ],
    },
    "喹硫平": {
        "en": "quetiapine",
        "urls": [
            "https://m.dayi.org.cn/drug/1152103",
            "http://s.dayi.org.cn/mip/s/medical/1152103.html",
        ],
    },
    "利培酮": {
        "en": "risperidone",
        "urls": [
            "https://m.dayi.org.cn/drug/1152029",
            "http://s.dayi.org.cn/mip/s/medical/1152029.html",
        ],
    },
    "阿立哌唑": {
        "en": "aripiprazole",
        "urls": [
            "https://m.dayi.org.cn/drug/1155811",
        ],
    },
    "氯氮平": {
        "en": "clozapine",
        "urls": [
            "https://m.dayi.org.cn/drug/1147029",
            "http://s.dayi.org.cn/mip/s/medical/1147029.html",
        ],
    },
    # ---------- 第四批：心境稳定剂 + 抗焦虑药 ----------
    "碳酸锂": {
        "en": "lithium",
        "urls": [
            "https://m.dayi.org.cn/drug/1156245",
            "http://s.dayi.org.cn/mip/s/medical/1156245.html",
        ],
    },
    "丙戊酸钠": {
        "en": "valproate",
        "urls": [
            "https://m.dayi.org.cn/drug/1152053",
            "http://s.dayi.org.cn/mip/s/medical/1152053.html",
        ],
    },
    "拉莫三嗪": {
        "en": "lamotrigine",
        "urls": [
            "https://m.dayi.org.cn/drug/1152307",
            "http://s.dayi.org.cn/mip/s/medical/1152307.html",
        ],
    },
    "丁螺环酮": {
        "en": "buspirone",
        "urls": [
            "https://m.dayi.org.cn/drug/1156721",
            "http://s.dayi.org.cn/mip/s/medical/1156721.html",
        ],
    },
}


def fetch_page(url: str) -> str | None:
    """抓取单个页面 HTML"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
    except Exception as e:
        print(f"    抓取失败: {e}")
        return None


def parse_dayi_page(html: str) -> dict:
    """从 dayi MIP/移动版页面中提取结构化信息"""
    soup = BeautifulSoup(html, "html.parser")
    info = {}

    # 尝试提取各个板块
    # dayi 的 MIP 页面通常有明确的分段标题（h2/h3）+ 段落

    # 1. 提取适应症
    for heading in soup.find_all(["h2", "h3", "h4"]):
        text = heading.get_text(strip=True)
        if re.search(r"适应[症证]|主治|功能", text):
            next_p = heading.find_next_sibling("p")
            if not next_p:
                next_div = heading.find_next("div")
                next_p = next_div.find("p") if next_div else None
            if next_p:
                info.setdefault("indications", []).append(next_p.get_text(strip=True))

    # 2. 提取所有可见文本（退一步策略）
    if not info:
        # 退一步：提取所有段落
        all_paragraphs = []
        for p in soup.find_all("p"):
            txt = p.get_text(strip=True)
            if txt and len(txt) > 10:
                all_paragraphs.append(txt)
        info["raw_paragraphs"] = all_paragraphs[:50]

        # 也尝试从所有 div 中拿文本
        all_divs = []
        for div in soup.find_all("div"):
            txt = div.get_text(strip=True)
            if txt and len(txt) > 30:
                all_divs.append(txt)
        info["raw_sections"] = all_divs[:30]

    # 3. 尝试提取结构化字段
    structured = {}
    all_text = soup.get_text(" ", strip=True)

    # 适应症段落
    for marker in ["适应症", "适应证", "主治功能", "功能主治"]:
        idx = all_text.find(marker)
        if idx > 0:
            chunk = all_text[idx:idx + 500]
            structured["indications_raw"] = chunk
            break

    # 用法用量段落
    for marker in ["用法用量", "用法与用量"]:
        idx = all_text.find(marker)
        if idx > 0:
            chunk = all_text[idx:idx + 800]
            structured["dosage_raw"] = chunk
            break

    # 不良反应
    for marker in ["不良反应", "副作用"]:
        idx = all_text.find(marker)
        if idx > 0:
            chunk = all_text[idx:idx + 1500]
            structured["adverse_raw"] = chunk
            break

    # 禁忌
    for marker in ["禁忌"]:
        idx = all_text.find(marker)
        if idx > 0 and all_text.find("禁忌症") < idx:
            continue
        idx2 = all_text.find("禁忌症")
        if idx2 > 0 and (idx < 0 or idx2 < idx):
            idx = idx2
        if idx > 0:
            chunk = all_text[idx:idx + 600]
            structured["contraindication_raw"] = chunk
            break

    info["structured"] = structured

    # 页面标题
    title_tag = soup.find("title")
    if title_tag:
        info["page_title"] = title_tag.get_text(strip=True)

    return info


def main():
    print("=== dayi.org.cn 精神科药物说明书抓取 ===\n")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_data = {}

    for drug_name, drug_info in DRUGS.items():
        print(f"[{drug_name}] 抓取中...")
        best_html = None
        best_url = None

        for url in drug_info["urls"]:
            print(f"  尝试: {url}")
            html = fetch_page(url)
            if html and len(html) > 2000:
                best_html = html
                best_url = url
                print(f"  成功 ({len(html)} 字节)")
                break
            else:
                print(f"  内容太短或失败")

        if not best_html:
            print(f"  所有 URL 均失败")
            all_data[drug_name] = {"error": "all urls failed"}
            continue

        # 保存原始 HTML
        raw_path = RAW_DIR / f"{drug_info['en']}.html"
        raw_path.write_text(best_html, encoding="utf-8")
        print(f"  原始 HTML 保存至: {raw_path}")

        # 解析
        info = parse_dayi_page(best_html)
        info["_fetched_url"] = best_url
        all_data[drug_name] = info

        time.sleep(1)  # 礼貌延迟

    # 保存到 JSON
    OUTPUT_JSON.write_text(
        json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结构化数据保存至: {OUTPUT_JSON}")
    print(f"原始 HTML 保存在: {RAW_DIR}")
    print("\n=== 完成 ===")
    print("请检查输出文件，确认关键数字（发生率、剂量范围）是否提取完整。")


if __name__ == "__main__":
    main()
