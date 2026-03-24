"""
Hallucination Detection Evaluation Framework
=============================================

This module provides tools for quantitative evaluation of the hallucination
detection system for research purposes.

Usage:
    python evaluation.py --test-file test_queries.json --output results.json
"""

import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from graph import app


@dataclass
class RelevancyMetrics:
    """Metrics for a single conversation/query."""
    query: str
    total_checks: int
    relevant_count: int
    irrelevant_count: int
    relevance_rate: float
    avg_review_duration: float
    avg_confidence: float
    high_confidence_rate: float  # % of checks with confidence >= 0.8
    agent_breakdown: Dict[str, Dict[str, int]]
    workflow_steps: int
    final_answer_relevant: bool
    timestamp: str


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    total_queries: int
    overall_relevance_rate: float
    avg_reviews_per_query: float
    agent_performance: Dict[str, Dict[str, float]]
    queries: List[RelevancyMetrics]
    evaluation_duration: float


class HallucinationEvaluator:
    """Evaluator for hallucination detection system."""
    
    def __init__(self):
        self.results: List[RelevancyMetrics] = []
    
    def run_single_evaluation(self, query: str, max_steps: int = 15) -> RelevancyMetrics:
        """Run evaluation on a single query."""
        initial_state = {
            "messages": [{"role": "user", "content": query}],
            "intent": "",
            "main_task": query,
            "research_findings": [],
            "draft": "",
            "critique_notes": "",
            "revision_number": 0,
            "next_step": "",
            "current_sub_task": "",
            "code_question": query,
            "code_snippet": "",
            "code_answer": "",
            "quiz_output": "",
            "relevancy_checks": [],
            "total_checks": 0,
            "relevant_count": 0,
            "irrelevant_count": 0,
            "agent_type_relevance": {}
        }
        
        config = {"recursion_limit": max_steps}
        final_state = None
        all_checks = []
        
        for step in app.stream(initial_state, config=config):
            node_name = list(step.keys())[0]
            node_state = step[node_name]
            final_state = node_state
            
            # Collect relevancy checks from each step
            if isinstance(node_state, dict) and node_state.get("relevancy_checks"):
                all_checks.extend(node_state["relevancy_checks"])
        
        # Calculate metrics
        total = final_state.get("total_checks", 0)
        relevant = final_state.get("relevant_count", 0)
        irrelevant = final_state.get("irrelevant_count", 0)
        
        # Calculate average review duration
        durations = [check.get("review_duration", 0) for check in all_checks]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Calculate confidence metrics
        confidences = [check.get("confidence", 0.5) for check in all_checks]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        high_confidence_count = sum(1 for c in confidences if c >= 0.8)
        high_confidence_rate = (high_confidence_count / len(confidences) * 100) if confidences else 0
        
        # Check if final answer was relevant
        final_answer_relevant = False
        if all_checks:
            # Look for the last writer or direct answer agent check
            for check in reversed(all_checks):
                if check.get("agent_name") in ["writer", "code_helper", "quiz_helper"]:
                    final_answer_relevant = check.get("is_relevant", False)
                    break
        
        return RelevancyMetrics(
            query=query,
            total_checks=total,
            relevant_count=relevant,
            irrelevant_count=irrelevant,
            relevance_rate=(relevant / total * 100) if total > 0 else 0,
            avg_review_duration=avg_duration,
            avg_confidence=avg_confidence,
            high_confidence_rate=high_confidence_rate,
            agent_breakdown=final_state.get("agent_type_relevance", {}),
            workflow_steps=total,
            final_answer_relevant=final_answer_relevant,
            timestamp=datetime.now().isoformat()
        )
    
    def run_batch_evaluation(self, queries: List[str], max_steps: int = 15) -> EvaluationReport:
        """Run evaluation on multiple queries."""
        start_time = time.time()
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Evaluating query {i}/{len(queries)}: {query[:50]}...")
            print('='*60)
            
            result = self.run_single_evaluation(query, max_steps)
            self.results.append(result)
            
            print(f"✓ Query completed: {result.relevance_rate:.1f}% relevance rate")
        
        evaluation_duration = time.time() - start_time
        return self._generate_report(evaluation_duration)
    
    def _generate_report(self, duration: float) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        if not self.results:
            raise ValueError("No evaluation results to report")
        
        # Overall statistics
        total_queries = len(self.results)
        total_relevant = sum(r.relevant_count for r in self.results)
        total_checks = sum(r.total_checks for r in self.results)
        overall_rate = (total_relevant / total_checks * 100) if total_checks > 0 else 0
        avg_reviews = total_checks / total_queries if total_queries > 0 else 0
        
        # Agent performance aggregation
        agent_stats: Dict[str, Dict[str, int]] = {}
        for result in self.results:
            for agent, stats in result.agent_breakdown.items():
                if agent not in agent_stats:
                    agent_stats[agent] = {"total": 0, "relevant": 0}
                agent_stats[agent]["total"] += stats.get("total", 0)
                agent_stats[agent]["relevant"] += stats.get("relevant", 0)
        
        # Calculate rates per agent
        agent_performance = {}
        for agent, stats in agent_stats.items():
            rate = (stats["relevant"] / stats["total"] * 100) if stats["total"] > 0 else 0
            agent_performance[agent] = {
                "total_checks": stats["total"],
                "relevant": stats["relevant"],
                "relevance_rate": round(rate, 2)
            }
        
        return EvaluationReport(
            total_queries=total_queries,
            overall_relevance_rate=round(overall_rate, 2),
            avg_reviews_per_query=round(avg_reviews, 2),
            agent_performance=agent_performance,
            queries=self.results,
            evaluation_duration=duration
        )
    
    def export_report(self, report: EvaluationReport, filepath: str):
        """Export report to JSON file."""
        # Convert dataclasses to dicts
        report_dict = {
            "total_queries": report.total_queries,
            "overall_relevance_rate": report.overall_relevance_rate,
            "avg_reviews_per_query": report.avg_reviews_per_query,
            "agent_performance": report.agent_performance,
            "evaluation_duration": report.evaluation_duration,
            "queries": [asdict(q) for q in report.queries]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Report exported to: {filepath}")
    
    def print_summary(self, report: EvaluationReport):
        """Print formatted summary to console."""
        print("\n" + "="*70)
        print("HALLUCINATION DETECTION EVALUATION REPORT")
        print("="*70)
        print(f"Total Queries Evaluated: {report.total_queries}")
        print(f"Overall Relevance Rate: {report.overall_relevance_rate:.1f}%")
        print(f"Avg Reviews per Query: {report.avg_reviews_per_query:.1f}")
        print(f"Evaluation Duration: {report.evaluation_duration:.1f}s")
        
        # Calculate and display confidence metrics
        avg_conf = sum(q.avg_confidence for q in report.queries) / len(report.queries) if report.queries else 0
        avg_high_conf = sum(q.high_confidence_rate for q in report.queries) / len(report.queries) if report.queries else 0
        print(f"\nConfidence Metrics:")
        print(f"  Average Confidence: {avg_conf:.2f}")
        print(f"  High Confidence Rate (≥0.8): {avg_high_conf:.1f}%")
        
        print("\n" + "-"*70)
        print("AGENT PERFORMANCE BREAKDOWN")
        print("-"*70)
        for agent, stats in sorted(report.agent_performance.items()):
            print(f"{agent:15} | Checks: {stats['total_checks']:3} | "
                  f"Relevant: {stats['relevant']:3} | Rate: {stats['relevance_rate']:5.1f}%")
        print("="*70)


def create_sample_test_set() -> List[str]:
    """Create a sample test set for evaluation."""
    return [
        "What is machine learning?",
        "Explain the concept of neural networks in detail.",
        "Write a Python function to calculate fibonacci numbers.",
        "Create a quiz about data structures.",
        "Compare supervised and unsupervised learning.",
        "How does backpropagation work?",
        "Generate a checklist for implementing a binary search tree.",
        "What are the advantages of using convolutional neural networks?"
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate hallucination detection system")
    parser.add_argument("--test-file", type=str, help="JSON file with test queries")
    parser.add_argument("--output", type=str, default="evaluation_report.json",
                        help="Output file for evaluation report")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Maximum workflow steps per query")
    
    args = parser.parse_args()
    
    # Load test queries
    if args.test_file:
        with open(args.test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            queries = test_data if isinstance(test_data, list) else test_data.get("queries", [])
    else:
        queries = create_sample_test_set()
        print("Using default test set with", len(queries), "queries")
    
    # Run evaluation
    evaluator = HallucinationEvaluator()
    report = evaluator.run_batch_evaluation(queries, args.max_steps)
    
    # Print and export results
    evaluator.print_summary(report)
    evaluator.export_report(report, args.output)
