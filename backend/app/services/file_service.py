import fitz  # PyMuPDF
from typing import List, Dict


def extract_text_by_page(pdf_path: str) -> List[Dict]:
    """
    从 PDF 中按页提取文本。

    Args:
        pdf_path: PDF 文件路径

    Returns:
        一个列表，每一项包含：
        - page: 页码（从 1 开始）
        - text: 当前页文本
    """
    results = []

    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()

            # 跳过空白页
            if not text:
                continue

            results.append(
                {
                    "page": page_index + 1,
                    "text": text,
                }
            )
    finally:
        doc.close()

    return results