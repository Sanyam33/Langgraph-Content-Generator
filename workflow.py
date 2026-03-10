from langgraph.graph import START, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,List,Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_groq import ChatGroq
import json, httpx, os


load_dotenv()

llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"), temperature=0)
# llm = ChatGroq(model=os.getenv("GROQ_MODEL"))

class LLMQuotaExceeded(Exception):
    pass

def handle_llm_error(e: Exception):
    msg = str(e)

    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        raise LLMQuotaExceeded("Gemini quota exceeded")

    raise e

class AgentState(TypedDict):

    # Inputs from the user
    website_url: str
    target_url: str
    anchor_text: str
    word_count: int
    category: str
    writing_style: str  # e.g., "Sales-driven", "Educational"
    
    # Internal workflow data
    scraped_content: str
    business_summary: str
    title_suggestions: List[str]
    seo_plan: str
    draft: str
    revision_notes: str
    is_approved: bool
    iteration_count: int

    scrape_success: bool
    error_message: str

# --- NODE 1: The Researcher ---
import requests
from bs4 import BeautifulSoup

async def research_node(state: AgentState):
    MAX_CONTENT_CHARS = 3000
    print(f"--- SCRAPING WEBSITE: {state['website_url']} ---")
    url = state['website_url']
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120 Safari/537.36"
        )
    }

    timeout = httpx.Timeout(10.0, connect=5.0)

    try:
        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            follow_redirects=True
        ) as client:

            response = await client.get(url)

            if response.status_code != 200:
                return {
                    "scraped_content": "",
                    "scrape_success": False,
                    "error_message": f"Failed to fetch content from {url}. Status code: {response.status_code}"
                }

            html = response.text

        soup = BeautifulSoup(html, "html.parser")

        # --- META DESCRIPTION ---
        meta_desc = ""
        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag:
            meta_desc = description_tag.get("content", "")

        # --- TITLE ---
        page_title = soup.title.string.strip() if soup.title and soup.title.string else "No Title Found"

        # --- REMOVE NOISE ---
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        raw_text = soup.get_text(separator=" ")
        clean_text = " ".join(raw_text.split())
        clean_text = clean_text[:MAX_CONTENT_CHARS]

        combined_content = (
            f"Title: {page_title}\n"
            f"Description: {meta_desc}\n\n"
            f"Content Snippet: {clean_text}"
        )

        return {
            "scraped_content": combined_content,
            "scrape_success": True,
            "error_message": ""
        }

    except httpx.ConnectTimeout:
        return {"scraped_content": f"Timeout while connecting to {url}", "scrape_success": False, "error_message": "Timeout while connecting to {url}"}

    except httpx.ReadTimeout:
        return {"scraped_content": f"Timeout while reading response from {url}", "scrape_success": False, "error_message": "Timeout while reading response from {url}"}

    except httpx.HTTPError as e:
        return {"scraped_content": f"HTTP error while fetching {url}: {str(e)}", "scrape_success": False, "error_message": f"HTTP error while fetching {url}: {str(e)}"}

    except Exception as e:
         return {
            "scraped_content": "",
            "scrape_success": False,
            "error_message": f"Scraping failed: {str(e)}"
        }

# --- NODE 2: The Analyst ---

def analyze_node(state: AgentState):
    print("--- ANALYZING BRAND VOICE & BUSINESS CONTEXT ---")
    
    # 1. Prepare the inputs from state
    content = state['scraped_content']
    style = state['writing_style']
    category = state['category']
    
    # 2. Define the Brand Strategist Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Brand Strategist. Your task is to analyze the scraped website content "
            "provided and create a concise Brand Profile. This profile will guide a writer to create "
            "consistent and effective content."
            "\n\n"
            "Focus on:\n"
            "1. Core Business Value: What do they actually sell or provide?\n"
            "2. Target Audience: Who are they talking to?\n"
            "3. Voice Adaptation: How should the requested writing style '{style}' be applied "
            "specifically for this {category} business?"
        )),
        ("human", "Website Scraped Content:\n\n{content}")
    ])
    chain = prompt | llm | StrOutputParser()

    # 4. Invoke the LLM
    try:
        brand_profile = chain.invoke({
            "content": content,
            "style": style,
            "category": category
        })
        
        print("✅ Brand Analysis Complete.")
        return {"business_summary": brand_profile}
    
    except Exception as e:
        error_msg = str(e)

        # ---- GEMINI QUOTA ----
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            raise LLMQuotaExceeded("Gemini API quota exceeded")

        print(f"❌ Analysis failed: {e}")
        return {"business_summary": f"Standard {category} profile with a {style} tone."}


# --- NODE 3: The SEO Architect ---
def plan_node(state: AgentState):
    print("--- PLANNING SEO & TITLES ---")
    
    # 1. Setup the prompt
    # We ask for JSON to easily separate the titles from the plan
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an SEO Content Architect. Your goal is to plan an article that naturally integrates a backlink."
            "\n\n"
            "Requirements:\n"
            "- Category: {category}\n"
            "- Target URL: {target_url}\n"
            "- Anchor Text: {anchor_text}\n"
            "- Word Count Goal: {word_count}\n"
            "- Style: {style}\n"
            "\n"
            "Tasks:\n"
            "1. Suggest 3 attention-grabbing titles.\n"
            "2. Create a structural outline (H1, H2, H3) that ensures we meet the word count.\n"
            "3. Specify exactly where the anchor text should be placed for maximum SEO value (e.g., middle of a specific section)."
            "\n\n"
            "Format your response as a JSON object with keys: 'titles' (a list) and 'outline' (a string)."
        )),
        ("human", "Business Summary to align with:\n\n{summary}")
    ])

    # 2. Create the chain
    chain = prompt | llm | JsonOutputParser()

    try:
        # 3. Invoke
        plan_output = chain.invoke({
            "category": state['category'],
            "target_url": state['target_url'],
            "anchor_text": state['anchor_text'],
            "word_count": state['word_count'],
            "style": state['writing_style'],
            "summary": state['business_summary']
        })
        
        print(f"SEO Planning Complete. Suggested Title: {plan_output['titles'][0]}")
        
        return {
            "title_suggestions": plan_output['titles'],
            "seo_plan": plan_output['outline']
        }

    except Exception as e:
        error_msg = str(e)
        # ---- GEMINI QUOTA ----
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            raise LLMQuotaExceeded("Gemini API quota exceeded")

        print(f"❌ SEO Planning failed: {e}")
        # Fallback if JSON parsing fails
        return {
            "title_suggestions": [f"Guide to {state['category']}"],
            "seo_plan": f"Standard outline for {state['word_count']} words. Place link on '{state['anchor_text']}'."
        }


# --- NODE 4: The Writer ---

def write_node(state: AgentState):
    print(f"--- WRITING CONTENT (Attempt {state.get('iteration_count', 0) + 1}) ---")
    
    # 1. Retrieve instructions and context from state
    style = state['writing_style']
    summary = state['business_summary']
    plan = state['seo_plan']
    title = state['title_suggestions'][0] # Using the primary suggested title
    target_url = state['target_url']
    anchor_text = state['anchor_text']
    word_count = state['word_count']
    
    # Check if there are revision notes from a previous loop
    notes = state.get('revision_notes', "This is the first draft. Follow the plan strictly.")

    # 2. Define the Writer Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a professional SEO Copywriter. Your task is to write a high-quality article that sounds 100% human and passes AI detection based on a provided plan"
            "\n\n"
            """
            CRITICAL HUMAN-TONE RULES:
            - NO AI CLICHÉS: Avoid words like 'delve', 'tapestry', 'unlock', 'harness', 'in the rapidly evolving landscape', 'comprehensive guide', or 'moreover'.
            - SENTENCE VARIETY: Use a mix of short, punchy sentences and longer, descriptive ones (Burstiness).
            - NO FLUFF: Get straight to the point. Use active voice. 
            - PLAGIARISM: Ensure all phrasing is original and unique."""
            "\n\n"
            "Strict Constraints:\n"
            "1. Tone: {style}\n"
            "2. Length: Approximately {word_count} words.\n"
            "3. SEO: You MUST include the anchor text '{anchor_text}' linked to '{target_url}' naturally within the body text.\n"
            "4. Brand Alignment: Follow this business profile: {summary}"
            "\n\n"
            "Current Feedback/Notes: {notes}"
        )),
        ("human", (
            "Write the article now.\n"
            "Title: {title}\n"
            "Outline/Plan to follow: {plan}"
        ))
    ])

    # 3. Create the chain
    chain = prompt | llm | StrOutputParser()

    try:
        # 4. Invoke the LLM to generate the draft
        content_draft = chain.invoke({
            "style": style,
            "word_count": word_count,
            "anchor_text": anchor_text,
            "target_url": target_url,
            "summary": summary,
            "notes": notes,
            "title": title,
            "plan": plan
        })
        
        print("✅ Content Drafted.")
        
        # We increment iteration_count here to track loops
        return {
            "draft": content_draft, 
            "iteration_count": state.get('iteration_count', 0) + 1
        }

    except Exception as e:
        error_msg = str(e)

        # ---- GEMINI QUOTA ----
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            raise LLMQuotaExceeded("Gemini API quota exceeded")

        print(f"❌ Writing failed: {e}")
        return {"draft": "Error in content generation."}


# --- NODE 5: The Editor (QC) ---
def review_node(state: AgentState):
    print("--- REVIEWING CONTENT (THE EDITOR) ---")

    # 1. Extract state variables upfront (fixes NameError on bare `draft` and `target_url`)
    draft: str = state["draft"]
    target_words: int = state["word_count"]
    anchor_text: str = state["anchor_text"]
    target_url: str = state["target_url"]
    writing_style: str = state["writing_style"]
    business_summary: str = state["business_summary"]

    actual_words: int = len(draft.split())

    # 2. Hard gate: HTML link presence check (before spending tokens on LLM review)
    has_html_link = (
        f'href="{target_url}"' in draft
        or f"href='{target_url}'" in draft
    )

    if not has_html_link:
        print("❌ Review failed: Missing HTML anchor tag.")
        return {
            "is_approved": False,
            "revision_notes": (
                "CRITICAL ERROR: The HTML hyperlink is missing. "
                f"You must wrap the anchor text in a proper tag: "
                f'<a href="{target_url}">{anchor_text}</a>. '
                "Do not bold or italicise — use the HTML <a> tag only."
            ),
        }

    # 3. Hard gate: word count floor check (before spending tokens on LLM review)
    lower_bound = int(target_words * 0.85)  # slightly tighter than the LLM's tolerance
    if actual_words < lower_bound:
        print(f"Review failed: Draft too short ({actual_words}/{target_words} words).")
        return {
            "is_approved": False,
            "revision_notes": (
                f"Draft is significantly too short ({actual_words} words). "
                f"The target is {target_words} words (minimum acceptable: ~{lower_bound}). "
                "Please expand the content before resubmitting."
            ),
        }

    # 4. LLM-based qualitative review
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            (
                "You are a Senior Content Editor. Evaluate whether the draft meets every client requirement below.\n\n"
                "Evaluation Criteria:\n"
                "1. Word Count - Is it approximately {target_words} words? (Actual: {actual_words}). "
                "A gap of ±30-50 words is acceptable.\n"
                "2. Anchor Text - Is '{anchor_text}' present in the text AND linked to '{target_url}' "
                "using a proper HTML <a> tag?\n"
                "3. Writing Style - Does the tone match '{style}'?\n"
                "4. Brand Alignment - Does the content reflect this business profile: {summary}?\n"
                "5. Human Quality - Does it read naturally, with no AI clichés or hollow filler phrases?\n\n"
                "Rules:\n"
                "- If ANY criterion fails, set is_approved to false.\n"
                "- revision_notes must always be a string (use an empty string '' if approved).\n"
                "- revision_notes must list every failing criterion with concrete, actionable fixes.\n\n"
                "Return ONLY a valid JSON object — no markdown, no explanation:\n"
                '{{"is_approved": true/false, "revision_notes": "..."}}'
            ),
        ),
        ("human", "Draft to Review:\n\n{draft}"),
    ])

    chain = prompt | llm | JsonOutputParser()

    try:
        review_result: dict = chain.invoke(
            {
                "target_words": target_words,
                "actual_words": actual_words,
                "anchor_text": anchor_text,
                "target_url": target_url,
                "style": writing_style,
                "summary": business_summary,
                "draft": draft,
            }
        )

        # Guarantee revision_notes is always a string (guards against None from LLM)
        review_result.setdefault("revision_notes", "")
        if review_result["revision_notes"] is None:
            review_result["revision_notes"] = ""

        # 5. Post-LLM word-count override (catches edge case where LLM was too lenient)
        # upper_bound = int(target_words * 1.15)
        # if actual_words > upper_bound:
        #     review_result["is_approved"] = False
        #     review_result["revision_notes"] += (
        #         f" Content is too long ({actual_words} words); "
        #         f"please trim to approximately {target_words} words."
        #     )

        print(f"Review complete. Approved: {review_result['is_approved']}")
        return {
            "is_approved": review_result["is_approved"],
            "revision_notes": review_result["revision_notes"].strip(),
        }

    except Exception as e:
        error_msg = str(e)
        # ---- GEMINI QUOTA ----
        if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
            raise LLMQuotaExceeded("Gemini API quota exceeded")
        # Fail-safe: reject rather than silently approve on parser/network errors
        print(f"Review node encountered an error: {e}")
        return {
            "is_approved": False,
            "revision_notes": (
                f"Review could not be completed due to an internal error: {e}. "
                "Please retry. The draft has NOT been approved automatically."
            ),
        }


def build_graph():
# Initialize the Graph
    workflow = StateGraph(AgentState)

    # 1. Add all your nodes
    workflow.add_node("researcher", research_node)
    workflow.add_node("analyze_node", analyze_node)
    workflow.add_node("planner", plan_node)
    workflow.add_node("writer", write_node)
    workflow.add_node("editor", review_node)

    # 2. Define the Linear Connections (Edges)
    workflow.add_edge(START, "researcher")      # Start here
    workflow.add_conditional_edges(
        "researcher",
        lambda state: "continue" if state.get("scrape_success") else "end",
        {
            "continue": "analyze_node",
            "end": END
        }
    )
    workflow.add_edge("analyze_node", "planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "editor")

    # 3. Define the Conditional Logic (The Loop)
    workflow.add_conditional_edges(
        "editor",
        lambda state: "rewrite" if not state["is_approved"] and state["iteration_count"] < 3 else "end",
        {
            "rewrite": "writer",
            "end": END
        }
    )

    # 4. Compile the Graph
    return workflow.compile()