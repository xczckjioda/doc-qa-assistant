import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)


EVALUATION_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVALUATION_DIR.parent
DATASET_PATH = EVALUATION_DIR / "eval_dataset.json"
RESULTS_PATH = EVALUATION_DIR / "eval_results.json"

sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from app.core.config import settings
from app.services.embedding_service import MODEL_NAME as EMBEDDING_MODEL_NAME
from app.services.retrieval_service import RetrievalService


def load_eval_dataset() -> List[Dict[str, str]]:
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("eval_dataset.json must contain a list of test cases.")

    for index, item in enumerate(data, start=1):
        if "question" not in item or "ground_truth" not in item:
            raise ValueError(
                f"Test case #{index} must contain question and ground_truth fields."
            )

    return data


def build_ragas_records(test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    retrieval_service = RetrievalService()
    records = []

    for item in test_cases:
        question = item["question"]
        ground_truth = item["ground_truth"]
        retrieval_output = retrieval_service.retrieve(query=question, top_k=5)

        contexts = [
            result["text"]
            for result in retrieval_output.get("results", [])
            if result.get("text")
        ]

        records.append(
            {
                "question": question,
                "answer": retrieval_output.get("answer", ""),
                "contexts": contexts,
                "ground_truth": ground_truth,
            }
        )

    return records


def build_evaluator_llm():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is required for RAGAS evaluation.")

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL_NAME,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def build_evaluator_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{EMBEDDING_MODEL_NAME}"
    )
    return LangchainEmbeddingsWrapper(embeddings)


def save_results(ragas_result, records: List[Dict[str, Any]], write_file: bool = True):
    result_df = ragas_result.to_pandas()
    metrics_mean = {}
    input_columns = {"question", "answer", "contexts", "ground_truth"}

    for column in result_df.columns:
        if column in input_columns:
            continue

        try:
            metrics_mean[column] = float(result_df[column].mean())
        except (TypeError, ValueError):
            pass

    output = {
        "metrics_mean": metrics_mean,
        "results": result_df.to_dict(orient="records"),
        "rag_records": records,
    }

    if write_file:
        with RESULTS_PATH.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    return output


def run_ragas_evaluation(
    test_cases: List[Dict[str, str]],
    write_file: bool = True,
) -> Dict[str, Any]:
    records = build_ragas_records(test_cases)

    dataset = Dataset.from_list(records)
    evaluator_llm = build_evaluator_llm()
    evaluator_embeddings = build_evaluator_embeddings()

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return save_results(ragas_result, records, write_file=write_file)


def main():
    test_cases = load_eval_dataset()
    output = run_ragas_evaluation(test_cases, write_file=True)

    print("RAGAS metric means:")
    for metric_name, value in output["metrics_mean"].items():
        print(f"{metric_name}: {value:.4f}")

    print(f"Full results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
