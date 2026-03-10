from pydantic import BaseModel

class ContentRequest(BaseModel):
    website_url: str
    target_url: str
    anchor_text: str
    word_count: int
    category: str
    writing_style: str

# --- 2. Response Schema ---
class TaskResponse(BaseModel):
    title:list[str] | None = None
    draft: str | None = None
    iteration_count: int = 0
    revision_notes: str | None = None
    error_message: str | None = None