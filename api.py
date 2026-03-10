
from fastapi import APIRouter, HTTPException
from schemas import ContentRequest, TaskResponse
from workflow import build_graph, LLMQuotaExceeded
import asyncio


router = APIRouter()
content_router = APIRouter(prefix="/api/v1", tags=["Content"])
graph = build_graph()  # compiled once

@router.post("/generate-content", response_model=TaskResponse, response_model_exclude_unset=True)
async def run_workflow(payload: ContentRequest):

    initial_state = {
        "website_url": payload.website_url,
        "target_url": payload.target_url,
        "anchor_text": payload.anchor_text,
        "word_count": payload.word_count,
        "category": payload.category,
        "writing_style": payload.writing_style,
        "iteration_count": 0
    }

    try:
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=60
        )
        # result = await graph.ainvoke(initial_state)

        # ---- SAFETY: Graph might return None in rare cases
        if not result:
            raise HTTPException(
                status_code=500,
                detail="Workflow execution returned no result"
            )

        # ---- SCRAPE FAILURE EARLY EXIT
        if not result.get("scrape_success", True):

            return TaskResponse(
                draft="No draft generated",
                iteration_count=0,
                error_message=result.get("error_message", "Website scraping failed")
            )

        # # ---- SAFE ACCESS
        # titles = result.get("title_suggestions", [])
        # title = titles[0] if titles else "Generated Article"

        return TaskResponse(
            title=result.get("title_suggestions",""),
            draft=result.get("draft", ""),
            iteration_count=result.get("iteration_count", 0),
            revision_notes=result.get("revision_notes", "")
        )

    except LLMQuotaExceeded:
        raise HTTPException(
            status_code=429,
            detail="AI generation failed because Gemini API quota has been exceeded. Please try again later."
        )

    except HTTPException:
        raise

    except Exception as e:

        print("Workflow Error:", error_msg)

        raise HTTPException(
            status_code=500,
            detail="Internal workflow execution error"
        )