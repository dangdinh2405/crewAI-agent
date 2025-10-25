# crew_setup.py
import os
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import BaseTool

# --- LLM setup (Gemini via env var) ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    # We'll raise in Flask route if missing, but keeping a fallback here too.
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=gemini_api_key,
    temperature=0.0,
)

# --- Tool ---
class CustomerSupportDataTool(BaseTool):
    name: str = "Customer Support Data Fetcher"
    description: str = (
        "Fetches recent customer support interactions, tickets, and feedback. Returns a summary string."
    )

    def _run(self, argument: str) -> str:
        # Simulated data fetch
        return (
            """Recent Support Data Summary:
- 50 tickets related to 'login issues'. High resolution time (avg 48h).
- 30 tickets about 'billing discrepancies'. Mostly resolved within 12h.
- 20 tickets on 'feature requests'. Often closed without resolution.
- Frequent feedback mentions 'confusing user interface' for password reset.
- High volume of calls related to 'account verification process'.
- Sentiment analysis shows growing frustration with 'login issues' resolution time.
- Support agent notes indicate difficulty reproducing 'login issues'."""
        )

support_data_tool = CustomerSupportDataTool()

# --- Agents ---
data_analyst = Agent(
    role="Customer Support Data Analyst",
    goal="Analyze customer support data to identify trends, recurring issues, and key pain points.",
    backstory=(
        "You are an expert data analyst specializing in customer support operations. "
        "Your strength lies in identifying patterns and quantifying problems from raw support data."
    ),
    verbose=False,
    allow_delegation=False,
    tools=[support_data_tool],
    llm=gemini_llm,
)

process_optimizer = Agent(
    role="Process Optimization Specialist",
    goal=(
        "Identify bottlenecks and inefficiencies in current support processes based on the data analysis. "
        "Propose actionable improvements."
    ),
    backstory=(
        "You are a specialist in optimizing business processes, particularly in customer support. "
        "You excel at pinpointing root causes of delays and suggesting concrete solutions."
    ),
    verbose=False,
    allow_delegation=False,
    llm=gemini_llm,
)

report_writer = Agent(
    role="Executive Report Writer",
    goal="Compile the analysis and improvement suggestions into a concise, clear, and actionable report for the COO.",
    backstory=(
        "You are a skilled writer adept at creating executive summaries and reports. "
        "You highlight the most critical information and recommendations for senior leadership."
    ),
    verbose=False,
    allow_delegation=False,
    llm=gemini_llm,
)

# --- Tasks ---
analysis_task = Task(
    description=(
        "Fetch and analyze the latest customer support interaction data (tickets, feedback, call logs) "
        "focusing on the last quarter. Identify the top 3-5 recurring issues, quantify their frequency "
        "and impact (e.g., resolution time, customer sentiment). Use the Customer Support Data Fetcher tool."
    ),
    expected_output=(
        "A summary report detailing: Top 3-5 recurring issues with frequency; average resolution times; "
        "key customer pain points from feedback; notable sentiment trends and agent observations."
    ),
    agent=data_analyst,
)

optimization_task = Task(
    description=(
        "Based on the data analysis report, identify the primary bottlenecks in the support processes "
        "contributing to the identified issues (especially the top recurring ones). Propose 2-3 concrete, "
        "actionable process improvements with impact and ease-of-implementation in mind."
    ),
    expected_output=(
        "A concise list linking main bottlenecks to key problems, plus 2-3 specific, actionable recommendations."
    ),
    agent=process_optimizer,
)

report_task = Task(
    description=(
        "Compile findings and recommendations into a 1-page executive report for the COO with sections: "
        "1) Critical issues (data points), 2) Process bottlenecks, 3) Actionable recommendations. "
        "Keep it clear, professional, and easy to scan."
    ),
    expected_output="A structured, concise executive report (max 1 page) with clear headings and bullet points.",
    agent=report_writer,
)

# --- Crew ---
support_analysis_crew = Crew(
    agents=[data_analyst, process_optimizer, report_writer],
    tasks=[analysis_task, optimization_task, report_task],
    process=Process.sequential,
    verbose=False,
)

def run_crew(data_query: str = "last quarter support data") -> str:
    """
    Executes the crew and returns the final COO report as a string.
    """
    result = support_analysis_crew.kickoff(inputs={"data_query": data_query})
    # Some Crew versions return a structured object; ensure it's a string:
    return str(result)

def run_agent1(data_query: str = "last quarter support data", feedback: str = "") -> str:
    """
    Chạy Agent đầu tiên (Data Analyst) với khả năng feedback
    """
    # Tạo task cho agent đầu tiên với feedback nếu có
    task_description = (
        "Fetch and analyze the latest customer support interaction data (tickets, feedback, call logs) "
        "focusing on the last quarter. Identify the top 3-5 recurring issues, quantify their frequency "
        "and impact (e.g., resolution time, customer sentiment). Use the Customer Support Data Fetcher tool."
    )
    
    if feedback:
        task_description += f"\n\nUser feedback: {feedback}\nPlease incorporate this feedback into your analysis."
    
    analysis_task_with_feedback = Task(
        description=task_description,
        expected_output=(
            "A summary report detailing: Top 3-5 recurring issues with frequency; average resolution times; "
            "key customer pain points from feedback; notable sentiment trends and agent observations."
        ),
        agent=data_analyst,
    )
    
    # Chạy chỉ agent đầu tiên
    result = analysis_task_with_feedback.execute()
    return str(result)

def run_agent5(data_query: str = "last quarter support data", 
               analysis_result: str = "", 
               optimization_result: str = "",
               feedback: str = "") -> str:
    """
    Chạy Agent thứ 5 (Report Writer) với khả năng feedback
    """
    # Tạo task cho agent cuối với feedback nếu có
    task_description = (
        "Compile findings and recommendations into a 1-page executive report for the COO with sections: "
        "1) Critical issues (data points), 2) Process bottlenecks, 3) Actionable recommendations. "
        "Keep it clear, professional, and easy to scan."
    )
    
    if feedback:
        task_description += f"\n\nUser feedback: {feedback}\nPlease incorporate this feedback into your report."
    
    if analysis_result:
        task_description += f"\n\nAnalysis results: {analysis_result}"
    
    if optimization_result:
        task_description += f"\n\nOptimization results: {optimization_result}"
    
    report_task_with_feedback = Task(
        description=task_description,
        expected_output="A structured, concise executive report (max 1 page) with clear headings and bullet points.",
        agent=report_writer,
    )
    
    # Chạy chỉ agent cuối
    result = report_task_with_feedback.execute()
    return str(result)