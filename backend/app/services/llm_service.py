import json
import re
from typing import List, Dict

from google import genai

from app.core.config import settings
from app.prompts.qa_prompt import QA_PROMPT_TEMPLATE


class LLMService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.LLM_MODEL_NAME

    def generate_answer(self, query: str, results: List[Dict]) -> Dict:
        if not results:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        context_parts = []
        raw_sources = []

        for item in results:
            text = item.get("text", "").strip()
            file_name = item.get("file_name", "unknown_file")
            page = item.get("page", None)

            if text:
                context_parts.append(f"[Source: {file_name}, page {page}]\n{text}")
                raw_sources.append({
                    "file_name": file_name,
                    "page": page
                })

        if not context_parts:
            return {
                "answer": "No relevant information found.",
                "sources": []
            }

        seen = set()
        sources = []

        for s in raw_sources:
            key = (s["file_name"], s["page"])
            if key not in seen:
                seen.add(key)
                sources.append(s)

        sources.sort(key=lambda x: (
            x["file_name"] or "",
            x["page"] if x["page"] is not None else 10**9
        ))

        context = "\n\n".join(context_parts)
        prompt = QA_PROMPT_TEMPLATE.format(query=query, context=context)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        answer_text = response.text.strip() if response.text else "No answer generated."

        return {
            "answer": answer_text,
            "sources": sources
        }

    def rewrite_query(self, query: str) -> str:
        prompt = f"""Rewrite the user question into a concise document retrieval query.
Preserve all specific entities, numbers, acronyms, and intent.
Do not answer the question.
Return only the rewritten query.

User question:
{query}
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        rewritten_query = response.text.strip() if response.text else ""
        return rewritten_query or query

    def generate_suggested_questions(self, query: str, answer: str) -> List[str]:
        prompt = f"""Based on the original question and answer, suggest exactly 3 concise follow-up questions.
Return only a JSON array of strings.

Original question:
{query}

Answer:
{answer}
"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        text = response.text.strip() if response.text else "[]"
        return self._parse_question_list(text)

    def _parse_question_list(self, text: str) -> List[str]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()][:3]
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed if str(item).strip()][:3]
            except json.JSONDecodeError:
                pass

        lines = [
            re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
            for line in text.splitlines()
        ]
        return [line for line in lines if line][:3]
