"""
Query router — the core research endpoint.

POST /api/query
  Request body:
    { "question": str, "mode": "explore" | "deep_dive" }

  Response: Server-Sent Events stream
    data: {"token": "..."}            — streamed answer text
    data: {"sources": [...]}          — source papers
    data: {"follow_up": [...]}        — follow-up questions (deep_dive only)
    data: {"confidence_note": "..."}  — disclaimer
    data: [DONE]                      — end of stream
"""
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.rag_pipeline import RAGPipeline

router = APIRouter()

# Lazy-initialize the RAG pipeline so the server starts fast.
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Return the singleton RAG pipeline, initializing on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    """Request schema for /api/query."""

    question: str = Field(..., min_length=3, max_length=2000)
    mode: str = Field("explore", pattern="^(explore|deep_dive)$")


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(data)}\n\n"


@router.post("/query")
async def query(req: QueryRequest):
    """Process a research query and stream the response via SSE."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_pipeline()

    async def event_stream():
        try:
            # Run retrieval
            retrieval_result = await pipeline.retrieve(req.question)

            # Send sources immediately
            if retrieval_result.sources:
                yield _sse({"sources": retrieval_result.sources})

            # Stream the Claude response token by token
            answer_parts = []
            async for chunk in pipeline.generate(
                context=retrieval_result.context,
                question=req.question,
                mode=req.mode,
            ):
                token = chunk.get("token", "")
                if token:
                    answer_parts.append(token)
                    yield _sse({"token": token})

            # In deep_dive mode, send follow-up questions
            if req.mode == "deep_dive":
                follow_ups = await pipeline.generate_follow_ups(
                    answer="".join(answer_parts),
                    question=req.question,
                )
                if follow_ups:
                    yield _sse({"follow_up": follow_ups})

            # Send confidence note
            today = datetime.now(timezone.utc).strftime("%B %d, %Y")
            yield _sse({
                "confidence_note": (
                    f"Based on available PMC open-access literature as of {today}. "
                    "Not medical advice."
                )
            })

            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield _sse({"error": str(exc)})
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
