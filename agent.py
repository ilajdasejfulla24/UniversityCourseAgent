from __future__ import annotations

from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END

from app.llm import get_llm
from app.rag import get_retriever


class RouteDecision(BaseModel):
    use_rag: bool = Field(description="Use internal course documents, prerequisites, requirements, advising FAQs.")
    use_web: bool = Field(description="Use web search for fresh information such as availability, policies, scholarships, job trends.")
    reason: str


class CourseAgentState(TypedDict, total=False):
    question: str
    route: RouteDecision
    rag_docs: List[Document]
    rag_answer: str
    web_results: str
    final_answer: str


@tool
def search_internal_course_documents(query: str) -> str:
    """Search internal university course documents and advising knowledge."""
    docs = get_retriever(k=5).invoke(query)
    return "\n\n".join(
        f"SOURCE: {doc.metadata.get('source', 'internal document')}\n{doc.page_content}"
        for doc in docs
    )


@tool
def search_web(query: str) -> str:
    """Search the public web for current course availability, policies, scholarships, or job trends."""
    return DuckDuckGoSearchRun().invoke(query)


def planner_node(state: CourseAgentState) -> CourseAgentState:
    llm = get_llm().with_structured_output(RouteDecision)
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a routing planner for a university course selection assistant.
Decide if the question needs: RAG over internal documents, web search for fresh information, or both.
Use RAG for course descriptions, prerequisites, degree requirements, workload, and advising FAQs.
Use web search for current availability, new policies, scholarships, internships, opportunities, or job market trends.
"""),
        ("human", "Question: {question}")
    ])
    decision = (prompt | llm).invoke({"question": state["question"]})
    return {**state, "route": decision}


def route_after_planner(state: CourseAgentState) -> Literal["rag", "web", "both"]:
    route = state["route"]
    if route.use_rag and route.use_web:
        return "both"
    if route.use_web:
        return "web"
    return "rag"


def rag_node(state: CourseAgentState) -> CourseAgentState:
    retriever = get_retriever(k=6)
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(
        f"[{i+1}] {doc.metadata.get('source', 'internal')}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an academic advising assistant. Use ONLY the internal course context below.
Extract relevant courses, prerequisites, workload notes, degree requirement information, and warnings.
Clearly say when internal documents do not contain enough information.
"""),
        ("human", "Student question: {question}\n\nInternal context:\n{context}")
    ])
    rag_answer = (prompt | get_llm()).invoke({"question": state["question"], "context": context}).content
    return {**state, "rag_docs": docs, "rag_answer": rag_answer}


def route_after_rag(state: CourseAgentState) -> Literal["web", "final"]:
    return "web" if state["route"].use_web else "final"


def web_search_node(state: CourseAgentState) -> CourseAgentState:
    query_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the student's question as a focused web search query. Include university/course/opportunity terms when useful."),
        ("human", "Question: {question}")
    ])
    web_query = (query_prompt | get_llm()).invoke({"question": state["question"]}).content
    results = search_web.invoke(web_query)
    return {**state, "web_results": results}


def final_node(state: CourseAgentState) -> CourseAgentState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful University Course Selection Assistant.
First understand the student's interests, year, completed prerequisites, workload needs, and career goals.
Then combine internal course information and web information.
Give clear course recommendations, prerequisite warnings, workload advice, and a short study plan.
You MUST label information sources using these headings:
- Internal university documents
- Web search / current information
- Recommendation
- Study plan
If a source was not used, say so briefly.
"""),
        ("human", """
Student question: {question}

Routing decision: {route_reason}

Internal RAG answer:
{rag_answer}

Web search results:
{web_results}
""")
    ])
    answer = (prompt | get_llm(temperature=0.2)).invoke({
        "question": state["question"],
        "route_reason": state.get("route").reason if state.get("route") else "No route recorded.",
        "rag_answer": state.get("rag_answer", "RAG was not used."),
        "web_results": state.get("web_results", "Web search was not used."),
    }).content
    return {**state, "final_answer": answer}


def build_graph():
    graph = StateGraph(CourseAgentState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("web_search_node", web_search_node)
    graph.add_node("final_node", final_node)

    graph.set_entry_point("planner_node")
    graph.add_conditional_edges(
        "planner_node",
        route_after_planner,
        {"rag": "rag_node", "web": "web_search_node", "both": "rag_node"},
    )
    graph.add_conditional_edges(
        "rag_node",
        route_after_rag,
        {"web": "web_search_node", "final": "final_node"},
    )
    graph.add_edge("web_search_node", "final_node")
    graph.add_edge("final_node", END)
    return graph.compile()
