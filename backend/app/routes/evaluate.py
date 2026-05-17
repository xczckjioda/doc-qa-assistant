from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from evaluation.ragas_eval import run_ragas_evaluation


router = APIRouter()


class EvaluateItem(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)


@router.post("/evaluate")
def evaluate_rag(test_cases: List[EvaluateItem]):
    try:
        payload = [
            item.model_dump() if hasattr(item, "model_dump") else item.dict()
            for item in test_cases
        ]
        output = run_ragas_evaluation(payload, write_file=True)
        metrics = output.get("metrics_mean", {})

        return {
            "success": True,
            "faithfulness": metrics.get("faithfulness"),
            "answer_relevancy": metrics.get("answer_relevancy"),
            "context_recall": metrics.get("context_recall"),
            "context_precision": metrics.get("context_precision"),
            "metrics": metrics,
            "results": output.get("results", []),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Evaluate failed: {str(e)}",
            "faithfulness": None,
            "answer_relevancy": None,
            "context_recall": None,
            "context_precision": None,
            "metrics": {},
            "results": [],
        }
