import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    清洗 PDF 提取后的文本，尽量减少不自然换行和多余空白。
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 先保留真正的段落边界：两个及以上换行
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # 把单个换行替换成空格，避免一句话被硬拆开
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 压缩多余空格和制表符
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def split_into_paragraphs(text: str) -> List[str]:
    """
    按段落切分：两个及以上换行视为段落分隔。
    """
    raw_paragraphs = re.split(r"\n\s*\n+", text)

    paragraphs = []
    for p in raw_paragraphs:
        cleaned = p.strip()
        if cleaned:
            paragraphs.append(cleaned)

    return paragraphs


def split_into_sentences(text: str) -> List[str]:
    """
    按中英文句末标点切句，并尽量保留标点。
    """
    text = text.strip()
    if not text:
        return []

    parts = re.split(r'([。！？；.!?])', text)

    sentences = []
    current = ""

    for part in parts:
        if not part:
            continue

        current += part

        if re.match(r'[。！？；.!?]', part):
            cleaned = current.strip()
            if cleaned:
                sentences.append(cleaned)
            current = ""

    if current.strip():
        sentences.append(current.strip())

    return sentences


def is_meaningful_chunk(text: str, min_chunk_length: int) -> bool:
    """
    过滤过短 chunk。
    """
    return len(text.strip()) >= min_chunk_length


def chunk_pages(
    pages: List[Dict],
    file_name: str,
    max_chunk_length: int = 500,
    min_chunk_length: int = 20,
) -> List[Dict]:
    """
    对每一页文本进行：
    1. 清洗
    2. 按段落切
    3. 段落过长时按句子边界切
    4. 再不行时退回硬切
    """
    chunks = []

    for page_item in pages:
        page_num = page_item["page"]
        text = clean_text(page_item["text"])

        if not text:
            continue

        paragraphs = split_into_paragraphs(text)
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 段落较短，直接作为一个 chunk
            if len(para) <= max_chunk_length:
                if is_meaningful_chunk(para, min_chunk_length):
                    chunks.append(
                        {
                            "chunk_id": f"{file_name}_page_{page_num}_chunk_{chunk_index}",
                            "file_name": file_name,
                            "page": page_num,
                            "text": para,
                        }
                    )
                    chunk_index += 1
                continue

            # 段落太长，按句子边界拼接
            sentences = split_into_sentences(para)
            current_chunk = ""

            for sentence in sentences:
                candidate = sentence if not current_chunk else current_chunk + " " + sentence

                if len(candidate) <= max_chunk_length:
                    current_chunk = candidate
                else:
                    if is_meaningful_chunk(current_chunk, min_chunk_length):
                        chunks.append(
                            {
                                "chunk_id": f"{file_name}_page_{page_num}_chunk_{chunk_index}",
                                "file_name": file_name,
                                "page": page_num,
                                "text": current_chunk.strip(),
                            }
                        )
                        chunk_index += 1

                    # 如果单句本身就太长，退回硬切
                    if len(sentence) > max_chunk_length:
                        start = 0
                        while start < len(sentence):
                            piece = sentence[start:start + max_chunk_length].strip()

                            if is_meaningful_chunk(piece, min_chunk_length):
                                chunks.append(
                                    {
                                        "chunk_id": f"{file_name}_page_{page_num}_chunk_{chunk_index}",
                                        "file_name": file_name,
                                        "page": page_num,
                                        "text": piece,
                                    }
                                )
                                chunk_index += 1

                            start += max_chunk_length

                        current_chunk = ""
                    else:
                        current_chunk = sentence

            # 收尾
            if is_meaningful_chunk(current_chunk, min_chunk_length):
                chunks.append(
                    {
                        "chunk_id": f"{file_name}_page_{page_num}_chunk_{chunk_index}",
                        "file_name": file_name,
                        "page": page_num,
                        "text": current_chunk.strip(),
                    }
                )
                chunk_index += 1

    return chunks