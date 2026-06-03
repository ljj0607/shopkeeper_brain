import re

def analyze_sections(md_path: str):
    """分析文档的 section 长度分布，辅助参数设定"""
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")
    sections = []
    current_title = ""
    body_lines = []
    in_fence = False

    def flush():
        body = "\n".join(body_lines).strip()
        if current_title or body:
            sections.append({"title": current_title, "body_len": len(body)})

    for line in content.split("\n"):
        if line.strip().startswith("```") or line.strip().startswith("~~~"):
            in_fence = not in_fence
        match = heading_re.match(line) if not in_fence else None
        if match:
            flush()
            current_title = line.strip()
            body_lines = []
        else:
            body_lines.append(line)
    flush()

    lengths = [s['body_len'] for s in sections]
    print(f"section 数: {len(sections)}")
    print(f"body 长度: 最小={min(lengths)}, 最大={max(lengths)}, "
          f"平均={sum(lengths)//len(lengths)}, 中位数={sorted(lengths)[len(lengths)//2]}")

    # 建议
    p75 = sorted(lengths)[int(len(lengths) * 0.75)]
    p25 = sorted(lengths)[int(len(lengths) * 0.25)]
    print(f"\n建议 max_content_length: {max(p75 * 2, 800)} ~ {max(p75 * 3, 1500)}")
    print(f"建议 min_content_length: {max(p25, 100)} ~ {max(p25 * 2, 200)}")