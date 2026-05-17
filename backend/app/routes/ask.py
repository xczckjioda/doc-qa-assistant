from fastapi import APIRouter, HTTPException
from app.schemas.query_schema import AskRequest, AskResponse
from app.services.retrieval_service import RetrievalService
import traceback

router = APIRouter()
retrieval_service = RetrievalService()


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        retrieval_output = retrieval_service.retrieve(
            query=request.query,
            top_k=request.top_k,
            file_name=request.file_name,
        )
        try:
            suggested_questions = retrieval_service.llm_service.generate_suggested_questions(
                query=request.query,
                answer=retrieval_output["answer"],
            )
        except Exception as e:
            print(f"Suggested question generation failed: {e}")
            suggested_questions = []

        return AskResponse(
            query=request.query,
            rewritten_query=retrieval_output.get("rewritten_query"),
            answer=retrieval_output["answer"],
            sources=retrieval_output["sources"],
            results=retrieval_output["results"],
            suggested_questions=suggested_questions,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ask failed: {str(e)}")
