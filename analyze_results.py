"""
Analysis Tools for Hallucination Detection Results
==================================================

Provides statistical analysis and visualization of evaluation results.

Usage:
    python analyze_results.py evaluation_report.json
"""

import json
import sys
from typing import Dict, List
from collections import defaultdict


def load_report(filepath: str) -> Dict:
    """Load evaluation report from JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_agent_consistency(queries: List[Dict]) -> Dict:
    """Analyze consistency of relevancy checks per agent type."""
    agent_checks = defaultdict(lambda: {"relevant": 0, "irrelevant": 0, "queries": []})
    
    for query in queries:
        for agent, stats in query.get("agent_breakdown", {}).items():
            agent_checks[agent]["relevant"] += stats.get("relevant", 0)
            agent_checks[agent]["irrelevant"] += stats.get("total", 0) - stats.get("relevant", 0)
            agent_checks[agent]["queries"].append(query["query"])
    
    # Calculate consistency metrics
    consistency = {}
    for agent, data in agent_checks.items():
        total = data["relevant"] + data["irrelevant"]
        consistency[agent] = {
            "total_checks": total,
            "relevance_rate": (data["relevant"] / total * 100) if total > 0 else 0,
            "consistency_score": 100 - (abs(data["relevant"] - data["irrelevant"]) / total * 50) if total > 0 else 0
        }
    
    return consistency


def analyze_query_complexity(queries: List[Dict]) -> Dict:
    """Analyze relationship between query characteristics and relevance."""
    analysis = {
        "short_queries": {"count": 0, "avg_relevance": 0},
        "long_queries": {"count": 0, "avg_relevance": 0},
        "high_relevance_queries": [],
        "low_relevance_queries": []
    }
    
    for query in queries:
        q_text = query["query"]
        relevance = query["relevance_rate"]
        
        # Categorize by length
        if len(q_text.split()) < 6:
            analysis["short_queries"]["count"] += 1
            analysis["short_queries"]["avg_relevance"] += relevance
        else:
            analysis["long_queries"]["count"] += 1
            analysis["long_queries"]["avg_relevance"] += relevance
        
        # Track extreme cases
        if relevance >= 80:
            analysis["high_relevance_queries"].append({"query": q_text, "rate": relevance})
        elif relevance <= 40:
            analysis["low_relevance_queries"].append({"query": q_text, "rate": relevance})
    
    # Calculate averages
    if analysis["short_queries"]["count"] > 0:
        analysis["short_queries"]["avg_relevance"] /= analysis["short_queries"]["count"]
    if analysis["long_queries"]["count"] > 0:
        analysis["long_queries"]["avg_relevance"] /= analysis["long_queries"]["count"]
    
    return analysis


def analyze_workflow_patterns(queries: List[Dict]) -> Dict:
    """Analyze workflow patterns and their impact on relevance."""
    patterns = {
        "avg_workflow_steps": 0,
        "steps_vs_relevance": [],
        "final_answer_success_rate": 0
    }
    
    total_final_relevant = 0
    for query in queries:
        steps = query.get("workflow_steps", 0)
        relevance = query.get("relevance_rate", 0)
        patterns["avg_workflow_steps"] += steps
        patterns["steps_vs_relevance"].append({"steps": steps, "relevance": relevance})
        
        if query.get("final_answer_relevant", False):
            total_final_relevant += 1
    
    if queries:
        patterns["avg_workflow_steps"] /= len(queries)
        patterns["final_answer_success_rate"] = (total_final_relevant / len(queries)) * 100
    
    return patterns


def generate_research_insights(report: Dict) -> List[str]:
    """Generate research insights from the evaluation."""
    insights = []
    queries = report.get("queries", [])
    
    # Overall performance
    overall_rate = report.get("overall_relevance_rate", 0)
    if overall_rate >= 70:
        insights.append(f"✓ Strong overall performance: {overall_rate:.1f}% relevance rate")
    elif overall_rate >= 50:
        insights.append(f"⚠ Moderate performance: {overall_rate:.1f}% relevance rate - room for improvement")
    else:
        insights.append(f"✗ Low performance: {overall_rate:.1f}% relevance rate - significant issues detected")
    
    # Agent-specific insights
    agent_perf = report.get("agent_performance", {})
    best_agent = max(agent_perf.items(), key=lambda x: x[1]["relevance_rate"]) if agent_perf else None
    worst_agent = min(agent_perf.items(), key=lambda x: x[1]["relevance_rate"]) if agent_perf else None
    
    if best_agent:
        insights.append(f"✓ Best performing agent: {best_agent[0]} ({best_agent[1]['relevance_rate']:.1f}%)")
    if worst_agent and worst_agent != best_agent:
        insights.append(f"⚠ Needs improvement: {worst_agent[0]} ({worst_agent[1]['relevance_rate']:.1f}%)")
    
    # Workflow insights
    patterns = analyze_workflow_patterns(queries)
    success_rate = patterns["final_answer_success_rate"]
    insights.append(f"📊 Final answer relevance rate: {success_rate:.1f}%")
    
    # Query complexity insights
    complexity = analyze_query_complexity(queries)
    if complexity["short_queries"]["count"] > 0 and complexity["long_queries"]["count"] > 0:
        short_avg = complexity["short_queries"]["avg_relevance"]
        long_avg = complexity["long_queries"]["avg_relevance"]
        if short_avg > long_avg:
            insights.append(f"📈 Short queries perform better ({short_avg:.1f}% vs {long_avg:.1f}%)")
        else:
            insights.append(f"📈 Long queries perform better ({long_avg:.1f}% vs {short_avg:.1f}%)")
    
    return insights


def print_detailed_analysis(report: Dict):
    """Print comprehensive analysis to console."""
    print("\n" + "="*80)
    print("DETAILED ANALYSIS REPORT")
    print("="*80)
    
    queries = report.get("queries", [])
    
    # Agent Consistency Analysis
    print("\n1. AGENT CONSISTENCY ANALYSIS")
    print("-"*80)
    consistency = analyze_agent_consistency(queries)
    for agent, metrics in sorted(consistency.items()):
        print(f"{agent:15} | Rate: {metrics['relevance_rate']:5.1f}% | "
              f"Consistency: {metrics['consistency_score']:5.1f}%")
    
    # Query Complexity Analysis
    print("\n2. QUERY COMPLEXITY ANALYSIS")
    print("-"*80)
    complexity = analyze_query_complexity(queries)
    print(f"Short queries (<6 words): {complexity['short_queries']['count']} "
          f"(avg relevance: {complexity['short_queries']['avg_relevance']:.1f}%)")
    print(f"Long queries (≥6 words):  {complexity['long_queries']['count']} "
          f"(avg relevance: {complexity['long_queries']['avg_relevance']:.1f}%)")
    
    if complexity["high_relevance_queries"]:
        print(f"\nHigh relevance queries (≥80%):")
        for q in complexity["high_relevance_queries"][:3]:
            print(f"  - {q['query'][:50]}... ({q['rate']:.1f}%)")
    
    if complexity["low_relevance_queries"]:
        print(f"\nLow relevance queries (≤40%):")
        for q in complexity["low_relevance_queries"][:3]:
            print(f"  - {q['query'][:50]}... ({q['rate']:.1f}%)")
    
    # Workflow Patterns
    print("\n3. WORKFLOW PATTERN ANALYSIS")
    print("-"*80)
    patterns = analyze_workflow_patterns(queries)
    print(f"Average workflow steps: {patterns['avg_workflow_steps']:.1f}")
    print(f"Final answer success rate: {patterns['final_answer_success_rate']:.1f}%")
    
    # Research Insights
    print("\n4. RESEARCH INSIGHTS")
    print("-"*80)
    insights = generate_research_insights(report)
    for insight in insights:
        print(f"  {insight}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <evaluation_report.json>")
        sys.exit(1)
    
    report_file = sys.argv[1]
    report = load_report(report_file)
    print_detailed_analysis(report)
