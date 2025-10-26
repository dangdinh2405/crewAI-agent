"""
crew_system.py
Skeleton multi-agent orchestration using CrewAI for the insurance price-suggest system.

How to use (quick):
    1) pip install crewai pydantic aiohttp matplotlib jinja2 weasyprint
       (We use simple libs; replace with your stack.)
    2) python crew_system.py
This will run a simulated end-to-end flow using mocked market data and RAG.
"""
import aiohttp, asyncio, json, re
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime
import google.generativeai as genai
import os
from pydantic import BaseModel, ValidationError
from crewai_tools import FileReadTool, SerperDevTool, ScrapeWebsiteTool
import csv
import matplotlib.pyplot as plt
from difflib import get_close_matches
from rapidfuzz import fuzz, process
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from dotenv import load_dotenv
import json as js
load_dotenv()

# WeasyPrint is optional on Windows because it requires native GTK/GObject libs
# which are often missing (error: cannot load library 'gobject-2.0-0').
# Try to import it, otherwise fall back to using Playwright for PDF rendering.
WEASYPRINT_AVAILABLE = False
try:
    from weasyprint import HTML, CSS  # type: ignore
    WEASYPRINT_AVAILABLE = True
except Exception:
    WEASYPRINT_AVAILABLE = False

# --- CrewAI imports (assumes crewai is installed) ---
# If your crewai version requires different names, adapt accordingly.
CREWAI_PRESENT = False
USE_NATIVE_CREW = os.getenv("USE_NATIVE_CREW", "0") == "1"
try:
    import crewai as _crewai_pkg
    # expose LLM type if available
    try:
        from crewai import Agent, Task, Crew, Process, LLM
    except Exception:
        # fallback: grab attributes from module if present
        Agent = getattr(_crewai_pkg, "Agent", None)
        Task = getattr(_crewai_pkg, "Task", None)
        Crew = getattr(_crewai_pkg, "Crew", None)
        Process = getattr(_crewai_pkg, "Process", None)
        LLM = getattr(_crewai_pkg, "LLM", None)
    CREWAI_PRESENT = True
except Exception:
    CREWAI_PRESENT = False

# By default we DO NOT use the native crewai Agent initializer because it will
# attempt to create LLM instances from environment (OPENAI_API_KEY etc.). Use
# USE_NATIVE_CREW=1 to force using the installed crewai classes.
if not (CREWAI_PRESENT and USE_NATIVE_CREW):
    # Provide local stub implementations so the script runs without needing
    # crewai native package or LLM environment variables.
    class Agent:
        def __init__(self, role: str, goal: str, backstory: str = "", handler=None, **kwargs):
            self.role = role
            self.goal = goal
            self.backstory = backstory
            self.handler = handler  # function to run

        async def run(self, input_data):
            if asyncio.iscoroutinefunction(self.handler):
                return await self.handler(input_data)
            return self.handler(input_data)

    class Task:
        def __init__(self, description: str = "", agent: Agent = None, **kwargs):
            self.description = description
            self.agent = agent

    class Process:
        sequential = "sequential"
        parallel = "parallel"

    class Crew:
        def __init__(self, agents: List[Agent], tasks: List[Task], process: str = Process.sequential, verbose: bool = True):
            self.agents = agents
            self.tasks = tasks
            self.process = process
            self.verbose = verbose

        async def kickoff(self, initial_input):
            context = {"session_input": initial_input, "agent_outputs": {}}
            for t in self.tasks:
                if self.verbose:
                    print(f"[Crew] Running task: {t.description} (agent={t.agent.role})")
                out = await t.agent.run(context)
                context["agent_outputs"][t.agent.role] = out
            return context

        async def run(self, input_data):
            if asyncio.iscoroutinefunction(self.handler):
                return await self.handler(input_data)
            return self.handler(input_data)

# --- End CrewAI fallback stubs ---

# Design API and tools
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
serper_tool = SerperDevTool(api_key=SERPER_API_KEY)
scrape_tool = ScrapeWebsiteTool()

# Tạo folder output nếu chưa tồn tại
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

async def _call_tool_flexible(tool, *a, **kw):
    """Try multiple calling signatures for tool.run to be robust against different tool implementations.

    Tries (in order):
      - tool.run(*a, **kw)
      - tool.run(a[0]) if single positional
      - tool.run(**kw)
      - tool.run()
    Returns the first successful result or raises the last exception.
    """
    last_exc = None
    # 1) try direct call
    try:
        out = tool.run(*a, **kw)
        if asyncio.iscoroutine(out):
            return await out
        return out
    except Exception as e:
        last_exc = e

    # 2) if there is exactly one positional arg, try passing it directly
    if len(a) == 1:
        try:
            out = tool.run(a[0])
            if asyncio.iscoroutine(out):
                return await out
            return out
        except Exception as e:
            last_exc = e

    # 3) try keyword-only
    if kw:
        try:
            out = tool.run(**kw)
            if asyncio.iscoroutine(out):
                return await out
            return out
        except Exception as e:
            last_exc = e

    # 4) try no-arg run()
    try:
        out = tool.run()
        if asyncio.iscoroutine(out):
            return await out
        return out
    except Exception as e:
        last_exc = e

    # nothing worked
    raise last_exc

# Configure Gemini
# def get_gemini_llm():
#     """
#     Returns a minimal LLM wrapper object compatible with CrewAI agent.llm usage.
#     We only define a 'run' method because handler functions usually call .run(prompt).
#     You can expand this later if needed.
#     """

#     gemini_llm = LLM(
#         model="gemini/gemini-2.5-flash",
#         api_key=GEMINI_API_KEY,
#         temperature=0.0,
#     )

#     return gemini_llm

def get_gemini_llm(model_name="gemini-2.5-flash-lite"):
    """
    Returns a minimal LLM wrapper object compatible with CrewAI agent.llm usage.
    We only define a 'run' method because handler functions usually call .run(prompt).
    You can expand this later if needed.
    """

    genai.configure(api_key=GEMINI_API_KEY)

    class GeminiLLMWrapper:
        def __init__(self, model):
            self.model = model

        def run(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.2):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens
                    }
                )
            except Exception as e:
                return f"[LLM Error] {str(e)}"

            # 1) Nếu response.text tồn tại → dùng ngay
            try:
                if hasattr(response, "text") and response.text:
                    return response.text.strip()
            except Exception:
                pass

            # 2) Nếu dùng candidates → tự extract text an toàn
            try:
                for c in response.candidates:
                    for part in c.content.parts:
                        if part.text:
                            return part.text.strip()
            except Exception:
                pass

            # 3) fallback cuối: trả JSON để debug
            try:
                return str(response)
            except:
                return "[No output returned]"
        
        # ---- CrewAI expects .predict() to exist ---
        def predict(self, prompt: str, **kwargs):
            return self.run(prompt, **kwargs)

        # ---- If called asynchronously ----
        async def predict_async(self, prompt: str, **kwargs):
            import asyncio
            return await asyncio.to_thread(self.run, prompt, **kwargs)

    model = genai.GenerativeModel(model_name)
    return GeminiLLMWrapper(model)

# -------------------------
# Schema / Types (simple)
# -------------------------

# Agent 1
class CustomerProfile(BaseModel):
    age: Optional[int]
    gender: Optional[Literal["Nam", "Nữ", "Khác"]] = None
    location: Optional[str] = None

class InsuranceRequest(BaseModel):
    policy_type: Optional[str]
    customer_profile: CustomerProfile
    benefits: List[str]
    price_hint: Optional[str]
    priority: Optional[Literal["cheap", "balanced", "benefit"]]

# Agent 2
class MarketItem(BaseModel):
    category: Optional[str]
    company_name: str
    product_name: str
    monthly_price: Optional[int]
    total_cost: Optional[int]
    description: Optional[str]
    eligibility: Optional[str]
    policy_id: str
    duration_years: Optional[int]

class MarketResult(BaseModel):
    count: int
    items: List[MarketItem]
    status: str

class StructuredRequest(BaseModel):
    policy_type: str  # mapped category, always valid
    customer_profile: CustomerProfile
    benefits: List[str]
    price_hint: Optional[str]
    priority: Optional[str]

    def normalize_policy_type(self) -> str:
        """
        Convert messy NLU string → standardized insurance category keyword
        """
        mapping = {
            "accident": ["tai nạn", "bảo hiểm tai nạn"],
            "health": ["sức khỏe", "bảo hiểm sức khỏe", "sk", "bhsk"],
            "life": ["nhân thọ", "bảo hiểm nhân thọ", "life"],
            "critical illness": ["hiểm nghèo", "bệnh hiểm nghèo"],
            "hospital": ["nằm viện", "trợ cấp viện phí", "hospital"],
            "children": ["trẻ em", "con nhỏ", "kid", "children"],
            "travel": ["du lịch", "travel"],
            "dental": ["nha khoa", "răng", "dental"],
            "vision": ["mắt", "kính", "vision"]
        }

        raw = (self.policy_type or "").lower()

        for key, keywords in mapping.items():
            for kw in keywords:
                if kw in raw:
                    return key

        # fallback: if can't understand → health
        return "health"

    def to_category(self) -> str:
        return self.normalize_policy_type()
    
# Agent 3
class CompanyProduct(BaseModel):
    category: Optional[str]
    company_name: str
    product_name: str
    monthly_price: Optional[int]
    total_cost: Optional[int]
    description: Optional[str]
    eligibility: Optional[str]
    policy_id: str
    duration_years: Optional[int]

# Agent 4
class WebMarketInsight(BaseModel):
    query_used: str
    references: List[str]
    highlights: List[str]  # các insight quan trọng về thị trường

# Agent 5
class AlternativeOption(BaseModel):
    action: str
    benefit: Optional[str] = None
    impact_profit_pct: Optional[float] = None

class EvaluatorOutput(BaseModel):
    recommended_price: Optional[int]
    current_company_price: Optional[int]
    market_median: Optional[int]
    delta_profit_pct: Optional[float]
    rationale: str
    alternatives: List[AlternativeOption]
    evaluated_at: str

# Agent 6
class AlternativeOption(BaseModel):
    action: str                  # e.g. "increase_price", "decrease_price", "add_benefit", "remove_benefit"
    detail: Optional[str] = None # e.g. "increase by 50k", "add outpatient coverage"
    impact_currency: Optional[float] = None  # estimated change in profit (VND)
    impact_pct: Optional[float] = None       # estimated change in profit (%)

class MarketSummary(BaseModel):
    market_median: Optional[float]
    market_mean: Optional[float]
    market_count: int

class CompanyMatch(BaseModel):
    product_id: Optional[str]
    name: Optional[str]
    current_price: Optional[float] # annual price in VND
    raw: dict

class EvaluatorResult(BaseModel):
    recommended_price: Optional[float]         # annual price in VND
    change_amount: Optional[float]             # recommended - current (VND)
    change_pct: Optional[float]                # change_amount / current * 100 (%)
    price_direction: Optional[str]             # "increase" / "decrease" / "keep"
    company: Optional[CompanyMatch]
    market_summary: MarketSummary
    benefits_to_add: List[str] = []
    benefits_to_remove: List[str] = []
    alternatives: List[AlternativeOption] = []
    rationale: str
    assumptions: List[str] = []
    evaluated_at: str

# Agent 7
class VisualizationArtifacts(BaseModel):
    price_chart_path: Optional[str] = None
    benefit_diff_chart_path: Optional[str] = None
    generated_at: str

# Agent 8
class ReportArtifact(BaseModel):
    report_html_path: Optional[str]   # sandbox:/mnt/data/... path
    report_md_path: Optional[str]
    report_pdf_path: Optional[str] = None
    generated_at: str

# -------------------------
# Helper / Mock Services
# -------------------------

# Agent 2
SCRAPER_URL = "https://api-data-insure.vercel.app/recommend"

TOP_N_LLM = 10
TOP_N_HEURISTIC = 10
TOP_N_FINAL = 15  # Số sản phẩm cuối cùng trả về

async def score_item_llm(item: MarketItem, struct_req: StructuredRequest, llm) -> MarketItem:
    prompt = f"""
Bạn là chuyên gia bảo hiểm. 
Khách hàng có yêu cầu: {struct_req.model_dump_json(indent=2)}
Sản phẩm hiện tại: {item.model_dump_json(indent=2)}

Đánh giá mức độ phù hợp sản phẩm với nhu cầu khách hàng 0–1.
"""
    try:
        if hasattr(llm, "predict_async"):
            score_text = await llm.predict_async(prompt)
        else:
            score_text = await asyncio.to_thread(llm.run, prompt)
        score = float(score_text.strip())
    except:
        score = 0.0
    item._llm_score = score
    return item

async def scrape_market_api_hybrid(struct_req: StructuredRequest, llm, top_n_final: int = TOP_N_FINAL) -> MarketResult:
    """
    Gọi API scrape và trả về top N sản phẩm theo kết hợp:
    1️⃣ LLM semantic match
    2️⃣ Heuristic + fuzzy match với benefits, price_hint, priority
    Merge kết quả, loại trùng policy_id và sắp xếp top N.
    """
    category_key = struct_req.to_category()
    payload = {"categories": [category_key]}

    # 1️⃣ Gọi API scrape
    async with aiohttp.ClientSession() as session:
        async with session.post(SCRAPER_URL, json=payload, timeout=30) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            if "application/json" not in content_type:
                error_text = await resp.text()
                print("⚠️ SCRAPER_URL trả về non-JSON:")
                print(error_text)
                return MarketResult(count=0, items=[], status="error_non_json_response")
            raw = await resp.json()

    items = [
        MarketItem(
            category=item.get("Category"),
            company_name=item.get("Company name"),
            product_name=item.get("Product Name"),
            monthly_price=int(item.get("Monthly") or 0),
            total_cost=int(item.get("Cost") or 0),
            description=item.get("Description"),
            eligibility=item.get("Eligibility"),
            policy_id=item.get("ID"),
            duration_years=int(item.get("Year") or 0)
        )
        for item in raw.get("items", [])
    ]

    # 2️⃣ LLM semantic scoring
    if llm is not None:
        llm_scored = await asyncio.gather(*[
            score_item_llm(item, struct_req, llm) for item in items
        ])
    else:
        # fallback: nếu không có llm → all score = 0
        for item in items:
            item._llm_score = 0.0
        llm_scored = items

    llm_scored.sort(key=lambda x: getattr(x, "_llm_score", 0), reverse=True)
    top_llm = llm_scored[:TOP_N_LLM]

    # 3️⃣ Heuristic + fuzzy match scoring
    heuristic_scored = []
    target_benefits = struct_req.benefits or []
    hint_value = None
    if struct_req.price_hint:
        try:
            hint_value = int(''.join(filter(str.isdigit, struct_req.price_hint)))
        except:
            hint_value = None

    for item in items:
        score = 0

        # Benefit match
        if target_benefits and item.description:
            desc_lower = item.description.lower()
            for b in target_benefits:
                if b.lower() in desc_lower:
                    score += 1

        # Fuzzy category match
        if category_key and item.category:
            score += int(fuzz.ratio(category_key.lower(), item.category.lower()) / 100)

        # Price hint
        if hint_value and item.monthly_price:
            diff_ratio = abs(item.monthly_price - hint_value) / hint_value
            if diff_ratio <= 0.2:
                score += 1

        # Priority
        if struct_req.priority:
            if struct_req.priority == "cheap" and item.monthly_price:
                score += max(0, 1 - (item.monthly_price / (hint_value or item.monthly_price)))
            elif struct_req.priority == "benefit":
                score += len(target_benefits)

        item._heuristic_score = score
        heuristic_scored.append(item)

    heuristic_scored.sort(key=lambda x: getattr(x, "_heuristic_score", 0), reverse=True)
    top_heuristic = heuristic_scored[:TOP_N_HEURISTIC]

    # 4️⃣ Merge hai danh sách, loại trùng policy_id
    merged_dict = {}
    for item in top_llm + top_heuristic:
        merged_dict[item.policy_id] = item  # tự động overwrite trùng id

    merged_list = list(merged_dict.values())
    merged_list.sort(
        key=lambda x: (getattr(x, "_llm_score", 0) + getattr(x, "_heuristic_score", 0)),
        reverse=True
    )

    top_items = merged_list[:top_n_final]

    return MarketResult(
        count=len(top_items),
        items=top_items,
        status=raw.get("status", "ok")
    )


# Agent 3
CSV_SUMMARY_PATH = "./data/products_expanded.csv"
csv_summary_tool = FileReadTool(file_path=CSV_SUMMARY_PATH)
CATEGORY_MAP: Dict[str, str] = {
    "life_insurance": "Life Insurance",
    "health_insurance": "Health Insurance",
    "accident_insurance": "Accident Insurance",
    "savings_plan": "Savings Plan",
    "retirement_plan": "Retirement Plan",
    "critical_illness": "Critical Illness"
}

async def rag_lookup_from_csv(struct_req: StructuredRequest, llm=None) -> List[CompanyProduct]:
    """
    Lookup sản phẩm công ty từ CSV:
    1. map category cứng
    2. fuzzy match
    3. LLM gợi ý nếu cần
    4. Lọc gần đúng theo benefits & price_hint từ NLU
    """
    category_key = struct_req.to_category()
    target_benefits = struct_req.benefits or []
    price_hint = struct_req.price_hint

    # Đọc CSV
    csv_categories = []
    rows = []
    with open(CSV_SUMMARY_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_categories.append(row["Category"])
            rows.append(row)

    # --- Step 1: Map cứng ---
    mapped_category = CATEGORY_MAP.get(category_key.lower())
    if mapped_category and mapped_category in csv_categories:
        matched_category = mapped_category
    else:
        # --- Step 2: Fuzzy match ---
        fuzzy_match = get_close_matches(category_key.replace("_", " ").lower(),
                                        [c.lower() for c in csv_categories], n=1, cutoff=0.6)
        if fuzzy_match:
            matched_category = fuzzy_match[0]
        else:
            # --- Step 3: LLM gợi ý ---
            if llm:
                prompt = f"""
Bạn là chuyên gia bảo hiểm. Khách hàng yêu cầu category: "{category_key}".
Các category CSV có sẵn: {csv_categories}.
Hãy trả về category phù hợp nhất.
"""
                if hasattr(llm, "predict_async"):
                    suggested_category = await llm.predict_async(prompt)
                else:
                    suggested_category = await asyncio.to_thread(llm.run, prompt)
                suggested_category = suggested_category.strip()
                fuzzy_from_llm = get_close_matches(suggested_category.lower(),
                                                  [c.lower() for c in csv_categories], n=1, cutoff=0.5)
                matched_category = fuzzy_from_llm[0] if fuzzy_from_llm else None
            else:
                matched_category = None

    if not matched_category:
        return []

    # --- Step 4: Lọc gần đúng theo benefits & price_hint ---
    filtered_rows = []
    for row in rows:
        if row["Category"] and row["Category"].strip().lower() == matched_category.lower():
            score = 0
            # Benefit matching
            if target_benefits and row.get("Description"):
                desc = row.get("Description", "").lower()
                for b in target_benefits:
                    if b.lower() in desc:
                        score += 1
            # Price hint matching (nếu có)
            if price_hint and row.get("Monthly"):
                try:
                    price_int = int(row.get("Monthly"))
                    hint_value = int(''.join(filter(str.isdigit, price_hint)))
                    # gần với hint thì tăng điểm
                    if abs(price_int - hint_value) / hint_value <= 0.3:  # +/-30%
                        score += 1
                except:
                    pass

            row["_match_score"] = score
            filtered_rows.append(row)

    # Sắp xếp theo điểm match giảm dần
    filtered_rows.sort(key=lambda r: r.get("_match_score", 0), reverse=True)

    # Chuyển thành Pydantic model
    results = [
        CompanyProduct(
            category=row.get("Category"),
            company_name=row.get("Company name"),
            product_name=row.get("Product Name"),
            monthly_price=int(row.get("Monthly") or 0),
            total_cost=int(row.get("Cost") or 0),
            description=row.get("Description"),
            eligibility=row.get("Eligibility"),
            policy_id=row.get("ID"),
            duration_years=int(row.get("Year") or 0)
        )
        for row in filtered_rows
    ]

    return results

# Agent 5
# helper: normalize a product dict -> annual price (VND)
def _annual_price_from_item(item: Dict[str, Any]) -> Optional[float]:
    # item may contain monthly_price, total_cost, Cost, Monthly, price, base_price...
    # check common fields in descending priority
    for key in ("monthly_price", "Monthly", "monthly", "price", "base_price", "total_cost", "Cost", "annual_price"):
        if key in item and item[key] is not None:
            try:
                v = float(item[key])
            except Exception:
                continue
            # if monthly-like keys -> multiply by 12
            if key.lower().startswith("monthly") or key.lower() == "monthly":
                return v * 12.0
            # if looks like monthly but numeric name "Monthly"
            if key == "Monthly":
                return v * 12.0
            # otherwise assume annual / total already
            return v
    return None

def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float((s[mid - 1] + s[mid]) / 2.0)

def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))

# -------------------------
# Agent Handlers
# -------------------------

# Agent 1 - NLU
async def handle_nlu(context: Dict[str, Any]) -> Dict[str, Any]:
    import json
    from datetime import datetime

    text = context.get("session_input", {}).get("text", "")
    llm = context.get("llm")

    if not text:
        return {"error": "no input text", "confidence": 0.0}

    prompt = f"""
Bạn là hệ thống NLU dành cho bảo hiểm.

Nhiệm vụ:
- Phân tích mô tả nhu cầu bảo hiểm của khách hàng (tiếng Việt)
- Trả về JSON có cấu trúc rõ ràng (không thêm giải thích)

Đầu vào:
{text}

Yêu cầu trả ra JSON đúng schema này:

{{
  "policy_type": "string",
  "customer_profile": {{
    "age": number,
    "gender": "Nam/Nữ/Khác",
    "location": "string (nếu có)"
  }},
  "benefits": ["danh sách quyền lợi"],
  "price_hint": "ước lượng giá nếu khách hàng có nhắc đến",
  "priority": "cheap / balanced / benefit"
}}
"""

    raw_output = llm.run(prompt)

    clean = raw_output.strip().replace("```json", "").replace("```", "").strip()

    try:
        data = js.loads(clean)
    except Exception as e:
        return {"error": f"JSON_PARSE_OR_SCHEMA_ERROR: invalid json: {e}", "raw_output": raw_output}

    # Try strict validation first; if nested optional fields are missing, try
    # to repair common omissions (e.g., missing customer_profile.location) and retry
    try:
        parsed = InsuranceRequest(**data)
    except ValidationError as ve:
        # Attempt to auto-fill commonly-missing optional nested fields and retry
        try:
            cust = data.get("customer_profile") if isinstance(data, dict) else None
            if cust is None or not isinstance(cust, dict):
                cust = {}
            # ensure optional keys exist with None defaults so pydantic will accept them
            if "age" not in cust:
                cust["age"] = None
            if "gender" not in cust:
                cust["gender"] = None
            if "location" not in cust:
                cust["location"] = None
            data["customer_profile"] = cust
            parsed = InsuranceRequest(**data)
        except Exception as e2:
            return {"error": f"JSON_PARSE_OR_SCHEMA_ERROR: {e2}", "raw_output": raw_output}
    except Exception as e:
        return {"error": f"JSON_PARSE_OR_SCHEMA_ERROR: {e}", "raw_output": raw_output}

    # ✅ convert to structured
    structured = StructuredRequest(
        policy_type=parsed.policy_type or "",
        customer_profile=parsed.customer_profile,
        benefits=parsed.benefits,
        price_hint=parsed.price_hint,
        priority=parsed.priority
    )

    standardized_category = structured.to_category()  # map to valid API type

    return {
        "structured_request": {
            **structured.model_dump(),
            "policy_type": standardized_category  # ✅ overwrite with mapped category
        },
        "confidence": 0.95,
        "parsed_at": datetime.utcnow().isoformat()
    }

# Agent 2 - SCRAPER
async def handle_scraper_hybrid(context: Dict[str, Any], agent_llm=None) -> Dict[str, Any]:
    """
    Agent 2 - SCRAPER Hybrid: tự động rank sản phẩm thị trường
    theo LLM + fuzzy + heuristic.
    """
    sr_dict = (
        context.get("agent_outputs", {}).get("NLU")
        or context.get("session_input", {}).get("structured_request")
    )
    if isinstance(sr_dict, dict) and "structured_request" in sr_dict:
        sr_dict = sr_dict["structured_request"]

    if not sr_dict:
        return {"error": "missing structured_request"}

    try:
        struct_req = StructuredRequest(**sr_dict)
    except Exception as e:
        return {"error": f"STRUCTURED_REQUEST_SCHEMA_ERROR: {e}", "raw": sr_dict}

    llm_to_use = agent_llm or get_gemini_llm()  # ✅ tạo default LLM
    # ✅ Sử dụng LLM được truyền từ Agent nếu có
    market: MarketResult = await scrape_market_api_hybrid(
        struct_req, 
        llm=llm_to_use,  # fallback nếu không có
        top_n_final=TOP_N_FINAL
    )

    return {
        "market_prices": market.model_dump(),
        "category_used": struct_req.to_category(),
        "fetched_at": datetime.utcnow().isoformat()
    }


# Agent 3 - RAG
async def handle_rag(context: Dict[str, Any], agent_llm=None) -> Dict[str, Any]:
    """
    Agent 3 - RAG Lookup: đối chiếu dataset nội bộ.
    Sử dụng LLM nếu cần cho việc mapping fuzzy + LLM.
    """
    sr_dict = (context.get("agent_outputs", {}).get("NLU")
            or context.get("session_input", {}).get("structured_request"))

    if isinstance(sr_dict, dict) and "structured_request" in sr_dict:
        sr_dict = sr_dict["structured_request"]

    if not sr_dict:
        return {"error": "missing structured request"}

    if isinstance(sr_dict, dict) and ("error" in sr_dict or "raw_output" in sr_dict):
        return {"error": "NLU_PARSE_ERROR", "details": sr_dict}

    try:
        struct_req = StructuredRequest(**sr_dict)
    except Exception as e:
        return {"error": f"STRUCTURED_REQUEST_INVALID: {e}"}

    company_products = await rag_lookup_from_csv(
        struct_req, 
        llm=agent_llm  # dùng LLM từ Agent nếu có
    )

    return {
        "company_products": [p.model_dump() for p in company_products],
        "retrieved_at": datetime.utcnow().isoformat(),
        "category_matched": struct_req.to_category()
    }


# Agent 4 - Web Market Search
async def handle_web_market_search(context: Dict[str, Any], agent_llm=get_gemini_llm(model_name="gemini-2.5-pro")) -> Dict[str, Any]:
    llm = agent_llm
    sr_dict = (context.get("agent_outputs", {}).get("NLU")
            or context.get("session_input", {}).get("structured_request"))

    if isinstance(sr_dict, dict) and "structured_request" in sr_dict:
        sr_dict = sr_dict["structured_request"]
    if not sr_dict:
        return {"error": "missing structured request"}

    struct_req = StructuredRequest(**sr_dict)

    current_year = datetime.now().year
    query = f"giá bảo hiểm {struct_req.to_category()} Vietnam cạnh tranh vào {current_year}"

    # 1) Search top 3 URLs
    try:
        search = await _call_tool_flexible(serper_tool, q=query)
        links = [r.get("link") for r in search.get("organic", [])[:3] if r.get("link")]
    except:
        links = []

    if not links:
        links = []

    # 2) Download all pages concurrently
    contents = await asyncio.gather(*[
        _call_tool_flexible(scrape_tool, url=url)
        for url in links
    ], return_exceptions=True)

    # Extract valid text only
    texts = [c for c in contents if isinstance(c, str) and len(c) > 200]

    # 3) If still no real data → fallback to synthetic market knowledge
    if not texts:
        synthetic = llm.run(f"""
Bạn là chuyên gia thị trường bảo hiểm Việt Nam.

Hãy tạo 5 insight cạnh tranh cho phân khúc bảo hiểm {struct_req.to_category()} gồm:
- Giá trung bình
- Giá cao / thấp phụ thuộc yếu tố nào
- Ưu điểm và nhược điểm của các gói thông dụng
- Thương hiệu dẫn đầu & phân khúc cạnh tranh
- Các yếu tố khách hàng thường quan tâm

Trả về dạng bullet point.
""")
        return {
            "web_market_insight": {
                "query_used": query,
                "references": links,
                "highlights": [x.strip("-• ") for x in synthetic.split("\n") if x.strip()]
            }
        }

    # 4) Summarize all pages together — only ONE LLM call!
    combined = "\n\n".join(texts)
    summary = llm.run(f"""
Tóm tắt nội dung thị trường sau và rút ra 5 insight cạnh tranh (dạng bullet):
{combined}
Và Trả về kết quả **chỉ dạng text thuần**, không JSON, không markdown, không meta-data.
Không được trả về bất kỳ thông tin hệ thống như: GenerateContentResponse, usage, candidates,...
""")

    clean = str(summary)
    clean = clean.replace("GenerateContentResponse", "").replace("response:", "")
    clean = clean.split("usage_metadata")[0]  # cắt metadata nếu lỡ xuất hiện

    highlights = [
        line.strip("-• ").strip()
        for line in clean.split("\n")
        if len(line.strip()) > 3 and not line.startswith("{")
    ]

    return {
        "web_market_insight": {
            "query_used": query,
            "references": links,
            "highlights": highlights
        }
    }


# Agent 5 - Evaluator
async def handle_evaluator(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluator: combine market_prices, company_products, web insights and structured_request
    Returns a dict matching EvaluatorResult schema.
    """
    print(context)
    outputs = context.get("agent_outputs", {})

    # 1) Extract inputs from context (support multiple naming variants)
    market_payload = (
        outputs.get("Scraper", {}) .get("market_prices")
        or outputs.get("MarketResearcher", {}).get("market_prices")
        or outputs.get("market_prices")
        or {}
    )
    company_payload = (
        outputs.get("RAG", {}).get("company_products")
        or outputs.get("CompanyScraper", {}).get("company_products")
        or outputs.get("company_products")
        or []
    )
    web_insight = outputs.get("MarketIntelligence", {}).get("web_market_insight") or outputs.get("web_market_insight")

    sr = outputs.get("NLU", {}).get("structured_request") or context.get("session_input", {}).get("structured_request")
    priority = (sr.get("priority") if isinstance(sr, dict) else None) or "balanced"

    # 2) Normalize market prices to annual numbers
    market_items = []
    if isinstance(market_payload, dict) and "items" in market_payload:
        market_items = market_payload.get("items", [])
    elif isinstance(market_payload, list):
        market_items = market_payload
    else:
        market_items = []

    market_ann_prices = []
    for item in market_items:
        p = _annual_price_from_item(item)
        if p is not None and p > 0:
            market_ann_prices.append(p)

     # 2b) Extract prices from web_insight highlights (look for VND numbers)
    web_prices = []
    if isinstance(web_insight, dict):
        for line in web_insight.get("highlights", []):
            matches = re.findall(r"([\d.,]+)\s*VND", line)
            for m in matches:
                try:
                    v = float(m.replace(",", ""))
                    if v > 0:
                        web_prices.append(v)
                except:
                    continue
    all_prices = market_ann_prices + web_prices

    market_median = _median(all_prices)
    market_mean = _mean(all_prices)
    market_count = len(all_prices)

    # 3) Choose company product match (prefer first)
    company_match = None
    current_company_price = None
    if isinstance(company_payload, list) and company_payload:
        cp = company_payload[0]
        current_company_price = _annual_price_from_item(cp) or cp.get("base_price") or cp.get("total_cost")
        company_match = {
            "product_id": cp.get("policy_id") or cp.get("ID") or cp.get("product_id"),
            "name": cp.get("product_name") or cp.get("name"),
            "current_price": current_company_price,
            "raw": cp
        }

    # 4) Determine recommended price (logic deterministic)
    recommended = None
    if current_company_price and market_median:
        if priority == "cheap":
            # aim to be slightly below market median, but not below 90% of current if current already cheaper
            target = market_median * 0.95
            recommended = min(current_company_price, target)
        elif priority == "benefit":
            # keep price, suggest improving benefits
            recommended = current_company_price
        else:  # balanced
            # move halfway towards market median
            recommended = current_company_price + 0.5 * (market_median - current_company_price)
            # round to nearest 1000 VND
            recommended = round(recommended / 1000.0) * 1000.0
    else:
        # fallback: prefer market_median if company price missing
        recommended = current_company_price or market_median or None

    # 5) compute changes & profit impact under simple assumption:
    # Assumption: profit per policy is proportional to price (i.e., fixed cost), so change in profit ~ change in price.
    # This is a simplifying assumption — we capture it in assumptions list.
    change_amount = None
    change_pct = None
    price_direction = "keep"
    if current_company_price and recommended is not None:
        # arithmetic digit-by-digit style (explicit)
        # change_amount = recommended - current_company_price
        change_amount = float(recommended) - float(current_company_price)
        if float(current_company_price) != 0:
            change_pct = (change_amount / float(current_company_price)) * 100.0
        if abs(change_amount) < 1e-6:
            price_direction = "keep"
        elif change_amount > 0:
            price_direction = "increase"
        else:
            price_direction = "decrease"

    # 6) Suggest benefits to add/remove using web_insight
    benefits_to_add = []
    benefits_to_remove = []

    if market_median and current_company_price:
        diff_pct_market = ((market_median - current_company_price) / market_median) * 100.0
        if diff_pct_market > 8.0 and priority != "cheap":
            benefits_to_add.extend(["ngoại trú", "mở rộng bảo hiểm ung thư (nếu chưa có)"])
        if ((current_company_price - market_median) / max(market_median,1)) * 100.0 > 10.0:
            benefits_to_remove.append("các rider ít sử dụng / chi phí cao")

    # 6b) parse extra benefits from web_insight lines (simple keyword heuristic)
    if isinstance(web_insight, dict):
        for line in web_insight.get("highlights", []):
            line_lower = line.lower()
            if "ngoại trú" in line_lower and "ngoại trú" not in benefits_to_add:
                benefits_to_add.append("ngoại trú")
            if "ung thư" in line_lower and "mở rộng bảo hiểm ung thư (nếu chưa có)" not in benefits_to_add:
                benefits_to_add.append("mở rộng bảo hiểm ung thư (nếu chưa có)")

    # 7) Alternatives (rough estimates)
    alternatives = []
    if current_company_price:
        alternatives.extend([
            {"action": "increase_price", "detail": "Increase price by 5%", "impact_currency": round(current_company_price * 0.05, 2), "impact_pct": 5.0},
            {"action": "decrease_price", "detail": "Decrease price by 5%", "impact_currency": round(-current_company_price * 0.05, 2), "impact_pct": -5.0},
        ])
        for b in benefits_to_add:
            alternatives.append({"action": "add_benefit", "detail": f"Add {b}", "impact_currency": round(-0.01 * current_company_price, 2), "impact_pct": -1.0})

    # Wrap alternatives
    alt_objs = [AlternativeOption(**a) for a in alternatives]

    # 8) Build rationale
    rationale_lines = [
        f"Market median annual price = {market_median if market_median is not None else 'N/A'} VND (based on {market_count} entries).",
        f"Company current annual price = {current_company_price if current_company_price is not None else 'N/A'} VND.",
        f"Customer priority = {priority}.",
        f"Recommended price direction = {price_direction}.",
    ]
    if change_amount is not None and change_pct is not None:
        rationale_lines.append(f"Recommended price change = {change_amount:+,.0f} VND ({change_pct:+.2f}%).")
    if benefits_to_add:
        rationale_lines.append(f"Suggested benefits to add: {', '.join(benefits_to_add)}.")
    if benefits_to_remove:
        rationale_lines.append(f"Suggested benefits to remove: {', '.join(benefits_to_remove)}.")

    rationale = " ".join(rationale_lines)

    assumptions = [
        "Assume annual price normalization: monthly * 12 if monthly given.",
        "Assume profit change is proportional to price change (fixed cost assumption).",
        "Estimates are rough; for exact profit impact use company-specific margin/cost model."
    ]

    result = EvaluatorResult(
        recommended_price=round(recommended, 2) if recommended is not None else None,
        change_amount=round(change_amount, 2) if change_amount is not None else None,
        change_pct=round(change_pct, 2) if change_pct is not None else None,
        price_direction=price_direction,
        company=CompanyMatch(**company_match) if company_match else None,
        market_summary=MarketSummary(
            market_median=round(market_median, 2) if market_median is not None else None,
            market_mean=round(market_mean, 2) if market_mean is not None else None,
            market_count=market_count
        ),
        benefits_to_add=benefits_to_add,
        benefits_to_remove=benefits_to_remove,
        alternatives=alt_objs,
        rationale=rationale,
        assumptions=assumptions,
        evaluated_at=datetime.utcnow().isoformat()
    )

    return result

# Agent 6 - Visualizer
async def handle_visualizer(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 6: Visualizer
    Input: EvaluatorResult (JSON)
    Output: Paths to generated charts
    """

    outputs = context.get("agent_outputs", {})
    eval_res = outputs.get("Evaluator") or context.get("session_input", {}).get("evaluator_result")

    artifacts = {
        "price_chart_path": None,
        "benefit_diff_chart_path": None,
        "generated_at": datetime.utcnow().isoformat()
    }

    if not eval_res:
        artifacts["error"] = "missing evaluator result"
        return artifacts

    try:
        result = EvaluatorResult(**eval_res)
    except Exception as e:
        artifacts["error"] = f"EVALUATOR_RESULT_INVALID: {e}"
        return artifacts

    # === 1) Price Comparison Chart ===
    try:
        fig, ax = plt.subplots()
        labels = ["Company Price", "Recommended Price", "Market Median"]
        values = [
            result.company.current_price if result.company else 0,
            result.recommended_price or 0,
            result.market_summary.market_median or 0
        ]
        ax.bar(labels, values)
        ax.set_ylabel("Annual Price (VND)")
        ax.set_title("Price Comparison: Company vs Recommendation vs Market")

        price_chart_path = os.path.join(output_dir, f"price_chart_{datetime.utcnow().timestamp()}.png")
        fig.savefig(price_chart_path, bbox_inches="tight")
        plt.close(fig)

        artifacts["price_chart_path"] = price_chart_path
    except Exception as e:
        artifacts["price_chart_error"] = str(e)

    # === 2) Benefit Change / Alternatives Chart ===
    try:
        if (result.benefits_to_add or result.benefits_to_remove):
            fig, ax = plt.subplots()
            labels = [f"+ {b}" for b in result.benefits_to_add] + [f"- {b}" for b in result.benefits_to_remove]
            values = [1] * len(labels)
            ax.barh(labels, values)
            ax.set_xlabel("Change Count")
            ax.set_title("Benefit Adjustments")

            benefit_chart_path = os.path.join(output_dir, f"benefit_chart_{datetime.utcnow().timestamp()}.png")
            fig.savefig(benefit_chart_path, bbox_inches="tight")
            plt.close(fig)

            artifacts["benefit_diff_chart_path"] = benefit_chart_path
    except Exception as e:
        artifacts["benefit_chart_error"] = str(e)

    artifacts["generated_at"] = datetime.utcnow().isoformat()
    return artifacts
# Agent 6 - Visualizer
async def handle_visualizer(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 6: Visualizer
    Input: EvaluatorResult (dict/JSON)
    Output: Paths to generated charts
    """
    outputs = context.get("agent_outputs", {})
    eval_res = outputs.get("Evaluator") or context.get("session_input", {}).get("evaluator_result")

    artifacts = {
        "price_chart_path": None,
        "benefit_diff_chart_path": None,
        "alternatives_chart_path": None,
        "generated_at": datetime.utcnow().isoformat()
    }

    if isinstance(eval_res, EvaluatorResult):
        result = eval_res
    else:
        try:
            result = EvaluatorResult(**eval_res)
        except Exception as e:
            artifacts["error"] = f"EVALUATOR_RESULT_INVALID: {e}"
            return artifacts

    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Price Comparison Chart ---
    try:
        fig, ax = plt.subplots()
        labels = ["Company Price", "Recommended Price", "Market Median"]
        values = [
            result.company.current_price if result.company else 0,
            result.recommended_price or 0,
            result.market_summary.market_median or 0
        ]
        colors = ["blue", "green", "orange"]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel("Annual Price (VND)")
        ax.set_title("Price Comparison: Company vs Recommendation vs Market")

        price_chart_path = os.path.join(output_dir, f"price_chart_{datetime.utcnow().timestamp()}.png")
        fig.savefig(price_chart_path, bbox_inches="tight")
        plt.close(fig)
        artifacts["price_chart_path"] = price_chart_path
    except Exception as e:
        artifacts["price_chart_error"] = str(e)

    # --- 2) Benefit Change Chart ---
    try:
        if result.benefits_to_add or result.benefits_to_remove:
            fig, ax = plt.subplots()
            labels = [f"+ {b}" for b in result.benefits_to_add] + [f"- {b}" for b in result.benefits_to_remove]
            values = [1] * len(labels)
            colors = ["green"] * len(result.benefits_to_add) + ["red"] * len(result.benefits_to_remove)
            ax.barh(labels, values, color=colors)
            ax.set_xlabel("Change Count")
            ax.set_title("Benefit Adjustments")

            benefit_chart_path = os.path.join(output_dir, f"benefit_chart_{datetime.utcnow().timestamp()}.png")
            fig.savefig(benefit_chart_path, bbox_inches="tight")
            plt.close(fig)
            artifacts["benefit_diff_chart_path"] = benefit_chart_path
    except Exception as e:
        artifacts["benefit_chart_error"] = str(e)

    # --- 3) Alternatives Chart ---
    try:
        if result.alternatives:
            fig, ax = plt.subplots()
            labels = [alt["action"] for alt in result.alternatives]
            values = [alt["impact_currency"] for alt in result.alternatives]
            colors = ["green" if v >= 0 else "red" for v in values]
            ax.bar(labels, values, color=colors)
            ax.set_ylabel("Impact (VND)")
            ax.set_title("Alternative Pricing Scenarios")

            alt_chart_path = os.path.join(output_dir, f"alternatives_chart_{datetime.utcnow().timestamp()}.png")
            fig.savefig(alt_chart_path, bbox_inches="tight")
            plt.close(fig)
            artifacts["alternatives_chart_path"] = alt_chart_path
    except Exception as e:
        artifacts["alternatives_chart_error"] = str(e)

    artifacts["generated_at"] = datetime.utcnow().isoformat()
    return VisualizationArtifacts(**artifacts).model_dump()

# Agent 7 - Report Generator
async def handle_report_generator(context: Dict[str, Any], agent_llm=None) -> Dict[str, Any]:
    """
    Agent 7: Compile structured request, evaluator result, and visualization charts into HTML/MD/PDF report.
    Enhanced version: includes CSS styling, structured tables, LLM narrative with retry/delay for quota limits.
    """
    import os
    from datetime import datetime
    import asyncio

    outputs = context.get("agent_outputs", {})

    nlp_out = outputs.get("NLU", {})
    sr_data = nlp_out.get("structured_request", {})
    ev_data = outputs.get("Evaluator", {}) or outputs.get("evaluator_result") or {}
    vis_data = outputs.get("Visualizer", {}) or outputs.get("visualization_artifacts") or {}

    # --- Safe parsing ---
    try:
        sr = InsuranceRequest(**sr_data) if sr_data else None
    except Exception:
        sr = None

    if isinstance(ev_data, EvaluatorResult):
        ev = ev_data
    else:
        try:
            ev = EvaluatorResult(**ev_data)
        except Exception:
            ev = None

    try:
        vis = VisualizationArtifacts(**vis_data) if vis_data else None
    except Exception:
        vis = None

    # --- Narrative via LLM with delay & retry per feature ---
    narrative = "Không có dữ liệu đầy đủ để tạo phân tích."
    llm = agent_llm or context.get("llm")

    if sr and ev and llm:
        # tạo prompt chi tiết dựa trên từng đặc tính
        summary_prompt = f"""Bạn là chuyên gia bảo hiểm cao cấp. Viết phân tích chi tiết & dễ hiểu về báo cáo giá bảo hiểm.
    Dữ liệu khách hàng:
    - Policy Type: {sr.policy_type}
    - Age: {sr.customer_profile.age}
    - Gender: {sr.customer_profile.gender}
    - Benefits: {', '.join(sr.benefits)}
    - Price Hint: {sr.price_hint}

    Kết quả định giá:
    - Company: {ev.company.name if ev.company else 'N/A'}
    - Current Price: {ev.company.current_price if ev.company else 'N/A'} VND
    - Recommended Price: {ev.recommended_price} VND
    - Change Amount: {ev.change_amount:+} VND
    - Change Percent: {ev.change_pct:+.2f}%
    - Price Direction: {ev.price_direction}
    - Benefits to Add: {', '.join(ev.benefits_to_add) if ev.benefits_to_add else 'None'}
    - Benefits to Remove: {', '.join(ev.benefits_to_remove) if ev.benefits_to_remove else 'None'}
    - Alternatives:
    """
        # if ev.alternatives:
        #     for alt in ev.alternatives:
        #         summary_prompt += f"  - Action: {alt.action}, Detail: {alt.detail}, Impact: {getattr(alt,'impact_currency',0):+,} VND ({getattr(alt,'impact_pct',0):+.2f}%)\n"

        summary_prompt += """
    Hãy viết nội dung giải thích thành dạng Markdown: tại sao giá đề xuất hợp lý, gợi ý hành động tiếp theo cho khách hàng, và tóm tắt chi tiết từng trường dữ liệu.
    """

        # --- Retry logic with dynamic delay for quota limits ---
        max_retries = 10  # tăng số lần thử
        narrative_done = False

        for attempt in range(1, max_retries + 1):
            try:
                await asyncio.sleep(50)  # delay nhỏ trước khi gọi LLM
                if hasattr(llm, "predict_async"):
                    resp = await llm.predict_async(summary_prompt, max_tokens=4096)
                    narrative = str(resp).strip()
                        
                else:
                    resp = await asyncio.to_thread(llm.predict, summary_prompt)
                    narrative = str(resp)
                narrative_done = True
                break
            except Exception as e:
                # kiểm tra nếu là quota exceeded
                retry_sec = retry_delay = 10  # default
                err_str = str(e)
                import re
                match = re.search(r"Please retry in (\d+\.?\d*)s", err_str)
                if match:
                    retry_sec = float(match.group(1))
                    print(f"[ReportGenerator] Quota exceeded, waiting {retry_sec:.1f}s before retry")
                else:
                    print(f"[ReportGenerator] LLM call failed, attempt {attempt}/{max_retries}: {e}")
                
                narrative = f"[LLM Error attempt {attempt}] {e}"
                await asyncio.sleep(retry_sec)

        if not narrative_done:
            narrative += " (LLM generation failed after retries)"

    # --- CSS ---
    css = """
    body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.5; }
    h1 { color: #2F4F4F; }
    h2, h3 { color: #4B0082; }
    table { border-collapse: collapse; width: 80%; margin-bottom: 20px; }
    th, td { border: 1px solid #999; padding: 8px; text-align: left; }
    th { background-color: #eee; }
    ul { margin: 5px 0 15px 20px; }
    img { margin: 10px 0; border: 1px solid #ccc; }
    footer { margin-top: 30px; font-size: 0.8em; color: #555; }
    """

    # --- Build HTML ---
    html = f"""
<html>
<head>
    <meta charset="utf-8">
    <title>Báo cáo đề xuất giá bảo hiểm</title>
    <style>{css}</style>
</head>
<body>
    <h1>📄 Báo cáo đề xuất giá bảo hiểm</h1>
"""

    # --- NLU Section ---
    if sr:
        html += "<h2>Thông tin khách hàng</h2>"
        html += f"<p>Policy Type: {sr.policy_type}</p>"
        customer = sr.customer_profile
        html += f"<p>Age: {getattr(customer,'age','')}, Gender: {getattr(customer,'gender','')}, Location: {getattr(customer,'location','')}</p>"
        html += f"<p>Benefits: {', '.join(sr.benefits)}</p>"
        html += f"<p>Price Hint: {sr.price_hint}</p>"
        html += f"<p>Priority: {sr.priority}</p>"

    # --- Evaluator Section ---
    if ev:
        html += "<h2>Kết quả đánh giá</h2>"
        if ev.company:
            html += f"<p>Product: {ev.company.name} ({ev.company.current_price:,} VND)</p>"
            html += f"<p>Category: {ev.company.raw.get('category','')}, Duration: {ev.company.raw.get('duration_years','')} years, Eligibility: {ev.company.raw.get('eligibility','')}</p>"

        html += f"<p>Recommended Price: {ev.recommended_price:,} VND</p>"
        html += f"<p>Price Change: {ev.change_amount:+,} VND ({ev.change_pct:+.2f}%), Direction: {ev.price_direction}</p>"

        if ev.rationale:
            html += "<p>Lý do & Phân tích:</p><ul>"
            for line in ev.rationale.split(". "):
                if line.strip():
                    html += f"<li>{line.strip()}</li>"
            html += "</ul>"

        if ev.assumptions:
            html += "<p>Giả định:</p><ul>"
            for a in ev.assumptions:
                html += f"<li>{a}</li>"
            html += "</ul>"

        # Alternatives Table
        if ev.alternatives:
            html += "<h3>Scenario Alternatives</h3>"
            html += "<table><tr><th>Action</th><th>Detail</th><th>Impact (VND)</th><th>Impact (%)</th></tr>"
            for alt in ev.alternatives:
                html += f"<tr><td>{getattr(alt,'action','')}</td><td>{getattr(alt,'detail','')}</td><td>{getattr(alt,'impact_currency',0):,}</td><td>{getattr(alt,'impact_pct',0):+.2f}%</td></tr>"
            html += "</table>"

        # Benefits Add/Remove
        if ev.benefits_to_add or ev.benefits_to_remove:
            html += "<h3>Benefit Adjustments</h3><ul>"
            for b in ev.benefits_to_add:
                html += f"<li>+ {b}</li>"
            for b in ev.benefits_to_remove:
                html += f"<li>- {b}</li>"
            html += "</ul>"

    # --- Visualizer Section ---
    if vis:
        html += "<h2>Biểu đồ & Dashboard</h2>"
        for chart in ["price_chart_path", "benefit_diff_chart_path", "alternatives_chart_path"]:
            path = getattr(vis, chart, None)
            if path:
                src_path = Path(path)
                web_path = Path(path).as_posix() 
                html += f'<img src="images/{src_path.name}" width="650"><br>'

    # --- Narrative Section ---
    # Convert markdown -> HTML
    narrative_html = markdown(narrative)
    html += f"<h2>Phân tích & Giải thích</h2>{narrative_html}"
    html += f"<footer><small>Generated at {datetime.utcnow().isoformat()}</small></footer></body></html>"

    # --- Save files ---
    base_dir = "./outputs"
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    html_path = os.path.join(base_dir, f"report_{ts}.html")
    md_path = os.path.join(base_dir, f"report_{ts}.md")

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(narrative)
    except Exception as e:
        print(f"[ReportGenerator] File write failed: {e}")

    # PDF generation if weasyprint available
    pdf_path = None
    if WEASYPRINT_AVAILABLE:
        try:
            from weasyprint import HTML
            pdf_path = os.path.join(base_dir, f"report_{ts}.pdf")
            HTML(string=html).write_pdf(pdf_path)
        except Exception as e:
            print(f"[ReportGenerator] WeasyPrint failed: {e}")

    return ReportArtifact(
        report_html_path=html_path if os.path.exists(html_path) else None,
        report_md_path=md_path if os.path.exists(md_path) else None,
        report_pdf_path=pdf_path if pdf_path and os.path.exists(pdf_path) else None,
        generated_at=datetime.utcnow().isoformat()
    ).model_dump()


# -------------------------
# Build Agents, Tasks, Crew
# -------------------------
def build_agents_and_crew(verbose=True):
    # LLM Gemini (dùng chung cho tất cả agent)
    gemini_llm = get_gemini_llm()

    # Agent objects
    agent_nlu = Agent(
        role="NLU",
        goal="Parse Vietnamese request into structured JSON",
        backstory="Sếp lớn NLU",
        handler=handle_nlu,
        llm=gemini_llm
    )

    agent_scraper = Agent(
        role="Scraper Hybrid",
        goal="Fetch top N market products using LLM + fuzzy + heuristic scoring",
        backstory="Tech1 - hybrid ranking agent",
        handler=handle_scraper_hybrid,
        llm=gemini_llm
    )

    agent_rag = Agent(
        role="RAG Lookup",
        goal="Truy xuất sản phẩm bảo hiểm nội bộ của công ty phù hợp với nhu cầu khách hàng",
        backstory="Tech2 - Truy vấn DB nội bộ + tài liệu sản phẩm",
        handler=handle_rag,
        llm=gemini_llm
    )

    agent_web = Agent(
        role="Market Intelligence Researcher",
        goal="Thu thập thông tin thị trường ngoài công khai để bổ trợ định giá",
        backstory="Chuyên gia phân tích thị trường bảo hiểm quốc gia & xu hướng cạnh tranh.",
        handler=handle_web_market_search,
        llm=get_gemini_llm(model_name="gemini-2.5-pro"),  # dùng model khác cho web search
        tools=[]  # ✅ đã áp dụng tool trong code
    )

    agent_eval = Agent(
        role="Evaluator",
        goal="Phân tích dữ liệu từ các agent khác để đưa ra giá đề xuất và lý do chi tiết.",
        backstory="Pricing engine chịu trách nhiệm đề xuất giá bán phù hợp dựa trên dữ liệu thị trường và công ty.",
        handler=handle_evaluator,
        llm=gemini_llm,
        allow_delegation=False,
    )

    agent_vis = Agent(
        role="Visualizer",
        goal="Produce charts and artifacts",
        backstory="Chart maker",
        handler=handle_visualizer,
        allow_code_execution=False,
        llm=None   # ✅ Không dùng LLM để sinh ảnh, tránh sinh text bậy
    )

    agent_report = Agent(
        role="ReportGenerator",
        goal="Compose structured HTML/PDF friendly report summarizing pricing analysis.",
        backstory="Tổng hợp nội dung từ toàn bộ pipeline và trình bày dễ hiểu.",
        handler=handle_report_generator,
        llm=gemini_llm,   # pass model-name or object depending on mode
    )

    # Tasks: here we arrange a logical pipeline. CrewAI supports flexible processes; we show sequential with parallel inside orchestrator.
    # We'll create tasks in the order we want them executed by our orchestrator.
    task_nlu = Task(
        name="Parse customer request",
        description="Chuyển mô tả nhu cầu bảo hiểm tiếng Việt thành JSON có cấu trúc phù hợp cho pipeline.",
        agent=agent_nlu,
        expected_output=InsuranceRequest.model_json_schema(),  # ✅ yêu cầu output đúng schema Pydantic
        output_key="structured_request"  # tên key trong kết quả handler trả ra
    )

    task_scraper = Task(
        name="Fetch top hybrid market products",
        description="Top N sản phẩm thị trường với hybrid scoring",
        agent=agent_scraper,
        expected_output=MarketResult.model_json_schema(),
        output_key="market_prices",
        context=[task_nlu]
    )
    
    task_rag = Task(
        name="Company product lookup",
        description="Dựa trên structured_request → lọc sản phẩm nội bộ theo category.",
        agent=agent_rag,
        expected_output=CompanyProduct.model_json_schema(),  # ✅ Output chuẩn schema
        output_key="company_products",
        context=[task_nlu]  # phụ thuộc task NLU trước đó
    )

    task_web = Task(
        name="Market Web Search",
        description="Tìm kiếm thông tin giá, xu hướng thị trường của loại bảo hiểm đã xác định.",
        agent=agent_web,
        expected_output=WebMarketInsight.model_json_schema(),
        output_key="web_market_insight",
        context=[task_nlu]  # phụ thuộc task NLU trước đó
    )

    task_evaluator = Task(
        description="Final evaluation task: merge market data and company data to recommend pricing.",
        agent=agent_eval,
        context=[
            task_nlu,                # Agent 1 output
            task_scraper,     # Agent 2 output
            task_rag,    # Agent 3 output
            task_web,      # Agent 4 output (Agent Search / serper + scrape)
        ],
        output_pydantic=EvaluatorOutput,  # chuẩn JSON trả về
        expected_output="Return recommended price JSON",
        human_input=True
    )

    task_vis = Task(
        name="Generate Visual Dashboard",
        description="Create price comparison and benefit difference charts from evaluator result.",
        agent=agent_vis,
        expected_output=VisualizationArtifacts.model_json_schema(),
        output_key="visualization_artifacts",
        context=[task_evaluator]
    )

    task_report = Task(
        name="Generate final pricing report",
        description="Compile structured request, evaluation results, and visual charts into a business report.",
        agent=agent_report,
        expected_output=ReportArtifact.model_json_schema(),
        output_key="report_artifact",
        context=[task_nlu, task_evaluator, task_vis]  # phụ thuộc 3 agent trước
    )
    
    crew = Crew(
        agents=[
            agent_nlu, agent_scraper, agent_rag, agent_web,
            agent_eval, agent_vis, agent_report
        ],
        tasks=[
            task_nlu, task_scraper, task_rag, task_web,
            task_evaluator, task_vis, task_report
        ],
        verbose=True
    )

    # Attach the actual LLM wrapper to the crew object so orchestrator/handlers
    # can access a .run()/.apredict() interface regardless of which Agent
    # representation was used for pydantic validation.
    try:
        setattr(crew, "_shared_llm", gemini_llm)
    except Exception:
        # best-effort; if Crew is immutable pydantic model, we'll rely on
        # orchestrator's fallback to search agents for llm or use None.
        pass

    return crew


# -------------------------
# Higher-level orchestrator using crew + parallel step
# -------------------------
async def orchestrator_run(crew: Crew, user_text: str):
    """
    Orchestrator that:
      - calls NLU (agent 1) synchronously
      - triggers Scraper & RAG in parallel (agents 2 & 3)
      - waits for both, then calls Evaluator -> Visualizer -> Report
    Returns aggregated context (agent outputs).
    """
    # 1) Run NLU
    initial_context = {"session_input": {"text": user_text}}
    print("[Orchestrator] Running NLU...")

    # Helper: find task by agent role name to avoid brittle index-based access
    def find_task_by_agent_role(role_name: str):
        for t in crew.tasks:
            agent_role = getattr(t.agent, "role", None)
            if agent_role == role_name:
                return t
        return None

    nlu_task = find_task_by_agent_role("NLU") or (crew.tasks[0] if crew.tasks else None)
    if nlu_task is None:
        raise RuntimeError("NLU task not found in crew.tasks")

    # Inject a shared llm into contexts if available on any agent
    # Prefer an explicit shared llm attached to the crew (set in build). Fallback
    # to finding an llm on agents (older behavior).
    shared_llm = getattr(crew, "_shared_llm", None)
    if shared_llm is None:
        for a in getattr(crew, "agents", []) or []:
            llm_candidate = getattr(a, "llm", None)
            if llm_candidate:
                shared_llm = llm_candidate
                break

    ctx_for_nlu = {"session_input": initial_context["session_input"], "llm": shared_llm}
    nlu_out = await nlu_task.agent.run(ctx_for_nlu)
    context = {"session_input": initial_context["session_input"], "agent_outputs": {"NLU": nlu_out}, "llm": shared_llm}

    # If NLU failed (returned an error dict), abort pipeline early and return NLU result
    if isinstance(nlu_out, dict) and ("error" in nlu_out or "raw_output" in nlu_out):
        print("[Orchestrator] NLU returned error, aborting pipeline.")
        return context

    # 2) Run Scraper & RAG in parallel
    print("[Orchestrator] Running Scraper and RAG in parallel...")
    scraper_task = find_task_by_agent_role("Scraper Hybrid")
    rag_task = find_task_by_agent_role("RAG Lookup") or find_task_by_agent_role("RAG")
    web_task = find_task_by_agent_role("Market Intelligence Researcher") or find_task_by_agent_role("Market Web Search")

    # prepare child contexts to hand to agents (they will find structured_request inside agent_outputs)
    child_context = {"agent_outputs": {"NLU": nlu_out}, "llm": shared_llm}

    coros = []
    if scraper_task:
        coros.append(scraper_task.agent.run(child_context))
    else:
        print("[Orchestrator] Warning: Scraper task not found; skipping")

    if rag_task:
        coros.append(rag_task.agent.run(child_context))
    else:
        print("[Orchestrator] Warning: RAG task not found; skipping")

    if web_task:
        coros.append(web_task.agent.run(child_context))
    else:
        print("[Orchestrator] Info: Web market search task not found; continuing without web insights")

    results = await asyncio.gather(*coros) if coros else []

    # assign results back based on which tasks existed
    ri = 0
    if scraper_task:
        context["agent_outputs"]["Scraper"] = results[ri]
        ri += 1
    if rag_task:
        context["agent_outputs"]["RAG"] = results[ri]
        ri += 1
    if web_task:
        web_out = results[ri]
        # store both under the agent role and a normalized key used by evaluator
        web_role = getattr(web_task.agent, "role", "Market Intelligence")
        context["agent_outputs"][web_role] = web_out
        # evaluator expects outputs.get("MarketIntelligence", {})
        context["agent_outputs"]["MarketIntelligence"] = web_out
        ri += 1

    # 3) Evaluator — find by role
    print("[Orchestrator] Running Evaluator...")
    eval_task = find_task_by_agent_role("Evaluator")
    if eval_task is None:
        # fallback: try to find a task with agent role containing 'Eval'
        for t in crew.tasks:
            ar = getattr(t.agent, "role", "")
            if "eval" in ar.lower():
                eval_task = t
                break

    if eval_task:
        eval_ctx = {**context, "llm": shared_llm}
        eval_out = await eval_task.agent.run(eval_ctx)
        context["agent_outputs"]["Evaluator"] = eval_out
    else:
        print("[Orchestrator] Warning: Evaluator task not found; skipping")

    # 4) Visualizer
    print("[Orchestrator] Running Visualizer...")
    vis_task = find_task_by_agent_role("Visualizer")
    if vis_task:
        vis_ctx = {**context}
        vis_out = await vis_task.agent.run(vis_ctx)
        context["agent_outputs"]["Visualizer"] = vis_out
    else:
        print("[Orchestrator] Warning: Visualizer task not found; skipping")

    # 5) Report
    print("[Orchestrator] Running ReportGenerator...")
    rep_task = find_task_by_agent_role("ReportGenerator") or find_task_by_agent_role("Report Generator")
    if rep_task:
        rep_ctx = {**context, "llm": shared_llm}
        rep_out = await rep_task.agent.run(rep_ctx)
        context["agent_outputs"]["ReportGenerator"] = rep_out
    else:
        print("[Orchestrator] Warning: ReportGenerator task not found; skipping")

    return context


# -------------------------
# API-style runner functions (mirror crew_setup.py style)
# Each function runs a single agent/task and returns a string result
# -------------------------
def _get_shared_crew_and_llm():
    """Helper: build crew and return (crew, shared_llm)."""
    crew = build_agents_and_crew(verbose=False)
    shared_llm = getattr(crew, "_shared_llm", None)
    if shared_llm is None:
        for a in getattr(crew, "agents", []) or []:
            llm_candidate = getattr(a, "llm", None)
            if llm_candidate:
                shared_llm = llm_candidate
                break
    return crew, shared_llm

def run_crew(data_query: str = "") -> str:
    """Run the full orchestrator pipeline and return JSON string of outputs.

    This mirrors crew_setup.run_crew but uses the local orchestrator.
    """
    crew = build_agents_and_crew(verbose=False)
    ctx = asyncio.run(orchestrator_run(crew, data_query))
    try:
        return json.dumps(ctx, ensure_ascii=False, default=str)
    except Exception:
        return str(ctx)


def run_agent1(data_query: str = "", feedback: str = "") -> str:
    """Run NLU agent (Agent 1). Accepts optional feedback. Returns JSON string."""
    crew, shared_llm = _get_shared_crew_and_llm()
    ctx = {"session_input": {"text": data_query}}
    if feedback:
        ctx["session_input"]["feedback"] = feedback
    ctx["llm"] = shared_llm
    out = asyncio.run(handle_nlu(ctx))
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except Exception:
        return str(out)


def run_agent2(data_query: str = "", analysis_result: str = "") -> str:
    """Run Scraper Hybrid (Agent 2).

    If analysis_result contains a structured_request JSON string it will be used,
    otherwise NLU will be run on data_query.
    Returns JSON string.
    """
    crew, shared_llm = _get_shared_crew_and_llm()

    struct = None
    if analysis_result:
        try:
            struct = analysis_result['structured_request']
        except Exception as e:
            print(e)
            struct = None

    if not struct:
        nlu_out = asyncio.run(handle_nlu({"session_input": {"text": data_query}, "llm": shared_llm}))
        if isinstance(nlu_out, dict):
            struct = nlu_out.get("structured_request")

    if not struct:
        return json.dumps({"error": "missing structured_request"}, ensure_ascii=False)

    out = asyncio.run(handle_scraper_hybrid({"session_input": {"structured_request": struct}, "llm": shared_llm}, agent_llm=shared_llm))
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except Exception:
        return str(out)


def run_agent3(data_query: str = "", analysis_result: str = "", optimization_result: str = "") -> str:
    """Run RAG lookup (Agent 3). Accepts structured_request JSON or raw text. Returns JSON string."""
    crew, shared_llm = _get_shared_crew_and_llm()

    struct = None
    if analysis_result:
        try:
            struct = analysis_result['structured_request']
        except Exception:
            struct = None

    if not struct:
        nlu_out = asyncio.run(handle_nlu({"session_input": {"text": data_query}, "llm": shared_llm}))
        if isinstance(nlu_out, dict):
            struct = nlu_out.get("structured_request")

    if not struct:
        return json.dumps({"error": "missing structured_request"}, ensure_ascii=False)

    out = asyncio.run(handle_rag({"session_input": {"structured_request": struct}, "llm": shared_llm}, agent_llm=shared_llm))
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except Exception:
        return str(out)


def run_agent4(data_query: str = "", analysis_result: str = "", optimization_result: str = "", additional_insights: str = "") -> str:
    """Run Web Market Search (Agent 4). Accepts structured_request JSON or raw text. Returns JSON string."""
    crew, shared_llm = _get_shared_crew_and_llm()

    struct = None
    if analysis_result:
        try:
            struct = analysis_result['structured_request']
        except Exception:
            struct = None

    if not struct:
        nlu_out = asyncio.run(handle_nlu({"session_input": {"text": data_query}, "llm": shared_llm}))
        if isinstance(nlu_out, dict):
            struct = nlu_out.get("structured_request")

    if not struct:
        return json.dumps({"error": "missing structured_request"}, ensure_ascii=False)

    out = asyncio.run(handle_web_market_search({"session_input": {"structured_request": struct}, "llm": shared_llm}, agent_llm=shared_llm))
   
    try:
        return json.dumps(out, ensure_ascii=False, default=str)
    except Exception:
        return str(out)


def run_agent5(data_query: str = "", analysis_result: str = "", optimization_result: str = "", additional_insights: str = "", qa_result: str = "", feedback: str = "") -> str:
    """Run Report Generator (Agent 5) and Evaluator. Accepts many optional JSON inputs (strings) and optional feedback.
    If nothing provided, runs the full crew. Returns JSON string.
    """
    crew, shared_llm = _get_shared_crew_and_llm()

    agent_outputs = {}
    if analysis_result:
        try:
            agent_outputs["NLU"] = {"structured_request": analysis_result["structured_request"]}
        except Exception as e:
            print(e)
            pass
    if optimization_result:
        try:
            agent_outputs["Scraper"] = optimization_result
        except Exception as e:
            print(e)
            pass
    if additional_insights:
        try:
            agent_outputs["RAG"] = additional_insights
        except Exception as e:
            print(e)
            pass
    if qa_result:
        try:
            agent_outputs["MarketIntelligence"] = qa_result
        except Exception as e:
            print(e)
            pass

    if not agent_outputs and data_query:
        return run_crew(data_query=data_query)

    ctx = {"agent_outputs": agent_outputs, "llm": shared_llm}
    if feedback:
        ctx["feedback"] = feedback

    # Run evaluator first
    eval_out = asyncio.run(handle_evaluator(ctx))
    
    # Add evaluator output to context for report generator
    if isinstance(eval_out, dict):
        agent_outputs["Evaluator"] = eval_out
        ctx["agent_outputs"] = agent_outputs

    # Then run report generator
    report_out = asyncio.run(handle_report_generator(ctx))
    
    try:
        # Return combined output
        return json.dumps({
            "evaluator": eval_out,
            "report": report_out
        }, ensure_ascii=False, default=str)
    except Exception:
        return str({
            "evaluator": eval_out,
            "report": report_out
        })
