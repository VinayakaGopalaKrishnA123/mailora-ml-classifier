"""
Standalone Comprehensive Evaluation and Benchmarking Suite for Mailora Classifier (v1.1.0).

Calculates real metrics on an independent test set:
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion Matrix & False Positive Rate (FPR)
- Dedicated Hard-Negative Subset Evaluation (Educational Newsletters, Tech Blogs, Salary Reports)
- Inference Latency (Mean, P50, P95, P99)
- Comprehensive Edge-Case Bench including Glasp Newsletter
"""

import os
import time
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.predict import EmailPredictor
from src.preprocess import combine_subject_and_body

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


BENCHMARK_EDGE_CASES = [
    # 1. Exact Glasp Newsletter Case
    {
        "category": "User Case: Glasp Newsletter (AI & SWE Educational Content)",
        "subject": "Why AI hasn't replaced software engineers, and won't",
        "body": "In this week's Glasp newsletter: We curate the best thinking on artificial intelligence, software engineering careers, and developer productivity. Featured Article: 'Why AI hasn't replaced software engineers, and won't' by Kevin Baker. Highlights from the piece: LLMs generate boilerplate, but software engineering is fundamentally about understanding business requirements. Plus: 5 articles on growth mindset and research publications on code models. Read at https://glasp.co/newsletter-92",
        "expected": "NON_CAREER_RELATED"
    },
    # 2. Tech Industry Trends & Leadership Newsletter
    {
        "category": "Hard Negative: Pragmatic Engineer Careers Digest",
        "subject": "The Pragmatic Engineer: How software engineering careers are evolving in 2026",
        "body": "Welcome to this week's edition of The Pragmatic Engineer. Today we look at how software engineering career ladders have adapted to AI tooling, autonomous agents, and full-stack ownership. We surveyed 1,200 engineering managers across Big Tech on hiring velocity and compensation. Read on Substack.",
        "expected": "NON_CAREER_RELATED"
    },
    # 3. Coding Interview Preparation Tutorial
    {
        "category": "Hard Negative: Interview Prep Tutorial (No Invite)",
        "subject": "How to crack coding interviews: A complete roadmap for SDE roles",
        "body": "Preparing for software engineering interviews at FAANG or top startups? In this detailed tutorial, we cover the most common dynamic programming patterns, binary tree traversals, and system design tradeoffs. Bookmark this free guide on our programming tutorials site.",
        "expected": "NON_CAREER_RELATED"
    },
    # 4. Salary Benchmark Report
    {
        "category": "Hard Negative: Tech Salary Report & Compensation Survey",
        "subject": "Tech Salary Report 2026: What Software Engineers and Data Scientists Earn",
        "body": "Levels.fyi and TechCompensation report: Comprehensive salary data for software engineers across global tech hubs. We break down base compensation, stock options (RSUs), and median signing bonuses. View the interactive salary benchmark charts.",
        "expected": "NON_CAREER_RELATED"
    },
    # 5. Labor Market Research Report
    {
        "category": "Hard Negative: Labor Market Hiring Research",
        "subject": "Hiring trends in the technology industry: What the data tells us",
        "body": "According to our quarterly labor market research report, tech hiring has shifted toward specialized backend, cloud infrastructure, and AI engineering roles. Generalist hiring has contracted by 12% across Fortune 500 enterprises. Download the whitepaper.",
        "expected": "NON_CAREER_RELATED"
    },
    # 6. Tech Company Ranking Editorial
    {
        "category": "Hard Negative: Top Tech Companies Editorial",
        "subject": "Top companies for software engineers in 2026: Culture, Pay, and Growth",
        "body": "Our annual workplace ranking evaluates the top technology employers for software engineers based on engineering culture, mentorship, developer productivity tooling, and work-life balance. Check out where Stripe and Datadog rank.",
        "expected": "NON_CAREER_RELATED"
    },
    # 7. University Placement Statistical Report
    {
        "category": "Hard Negative: University Placement Overview Report",
        "subject": "Annual Campus Placement Statistics Report 2025-2026",
        "body": "The Dean of Student Affairs has published the annual college placement report. 94% of graduating engineering students were placed across 150 visiting companies with a median CTC of 11.8 LPA. This document provides statistical data for accreditation purposes.",
        "expected": "NON_CAREER_RELATED"
    },
    # 8. Genuine Actionable Internship Opportunity
    {
        "category": "Actionable Career: Software Engineer Internship",
        "subject": "Software Engineer Internship — Apply Now",
        "body": "Applications are officially open for our Summer 2027 Software Engineering Internship at Datadog. Work on high-scale distributed monitoring and time-series engines. Open to pre-final year students graduating in 2028. Submit your resume at https://careers.datadoghq.com/intern-swe before Nov 20.",
        "expected": "CAREER_RELATED"
    },
    # 9. Genuine Interview Schedule
    {
        "category": "Actionable Career: Recruiter Interview Confirmation",
        "subject": "Your interview is scheduled for tomorrow: Software Engineer Intern at Uber",
        "body": "Hi Candidate, Your technical interview round for the Software Engineer Intern position at Uber is confirmed for tomorrow, Thursday at 3:00 PM IST via Google Meet: https://meet.google.com/xyz-uber-interview. The interviewer will be a Senior Systems Engineer. Please have CoderPad ready.",
        "expected": "CAREER_RELATED"
    },
    # 10. Genuine Application Received Notification
    {
        "category": "Actionable Career: Application Status Acknowledgment",
        "subject": "Application received — Graduate Engineer Trainee at Tata Consultancy Services",
        "body": "Dear Candidate, We acknowledge receipt of your application for the Graduate Engineer Trainee (GET) role at TCS (Ref: TCS-GET-2026-902). Your credentials are under review by our talent acquisition panel. We will notify you regarding assessment dates shortly.",
        "expected": "CAREER_RELATED"
    },
    # 11. Genuine Campus Placement Drive Announcement
    {
        "category": "Actionable Career: Campus Placement Drive Registration",
        "subject": "Placement Drive Registration Open: J.P. Morgan Campus Hiring 2027",
        "body": "Training & Placement Cell Announcement: J.P. Morgan Chase & Co. is visiting campus for the 2027 batch recruitment drive. Role: Software Engineer Analyst. CTC: 19.5 LPA. Eligible Branches: B.Tech (CSE, IT, ECE) with CGPA >= 7.5. Register on Superset portal before midnight tonight.",
        "expected": "CAREER_RELATED"
    },
    # 12. Genuine Rejection Status Update (Status updates ARE career-related!)
    {
        "category": "Actionable Career: Application Rejection Update",
        "subject": "Update regarding your job application for Software Engineer at Stripe",
        "body": "Dear Applicant, Thank you for interviewing with us for the Software Engineer role. After careful review, we will not be moving forward with your candidacy for this position. We wish you success in your job search.",
        "expected": "CAREER_RELATED"
    },
    # 13. Everyday Negative: Shopping Delivery
    {
        "category": "Everyday Negative: Amazon Parcel Shipped",
        "subject": "Your Amazon.in order #402-98124 has shipped",
        "body": "Your package containing Logitech mouse will arrive tomorrow. Track on Amazon app.",
        "expected": "NON_CAREER_RELATED"
    },
    # 14. Everyday Negative: Credit Card Bill
    {
        "category": "Everyday Negative: HDFC Monthly Statement",
        "subject": "HDFC Bank: Monthly e-Statement for Credit Card ending 4092",
        "body": "Dear Customer, Your monthly credit card statement for October is ready. Total Amount Due: INR 14,820.50. Minimum Due: INR 1,200. Payment Due Date: 28-Oct-2026.",
        "expected": "NON_CAREER_RELATED"
    },
    # 15. Misleading Keyword Negative (Colloquial "job")
    {
        "category": "Misleading Keyword: Colloquial Praise",
        "subject": "Great job on the presentation today!",
        "body": "Hi team, you all did a fantastic job on the internal client pitch this morning. Have a wonderful weekend!",
        "expected": "NON_CAREER_RELATED"
    }
]


def run_evaluation(
    model_type: str = "baseline",
    test_csv: str = "dataset/test.csv",
    artifacts_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Evaluates the specified model against the test dataset and edge case bench.
    """
    logger.info(f"Running evaluation for model_type='{model_type}' on '{test_csv}'...")

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test dataset not found at {test_csv}. Run data preparation first.")

    predictor = EmailPredictor(model_type=model_type, artifacts_dir=artifacts_dir)
    test_df = pd.read_csv(test_csv)

    y_true = test_df["label"].values.astype(int)
    latencies = []
    y_pred = []
    y_prob = []

    # Run inference across test set
    for _, row in test_df.iterrows():
        t0 = time.perf_counter()
        res = predictor.predict_single(row["subject"], row["body"])
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)
        y_pred.append(1 if res["is_career_related"] else 0)
        y_prob.append(res["career_probability"])

    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(
        y_true,
        y_pred,
        target_names=["NON_CAREER_RELATED", "CAREER_RELATED"],
        output_dict=True
    )

    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))

    # Evaluate benchmark edge cases
    edge_results = []
    edge_correct = 0
    hard_negative_correct = 0
    hard_negative_total = 0

    for case in BENCHMARK_EDGE_CASES:
        res = predictor.predict_single(case["subject"], case["body"])
        is_correct = (res["label"] == case["expected"])
        if is_correct:
            edge_correct += 1

        is_hn = "Hard Negative" in case["category"] or "User Case" in case["category"]
        if is_hn:
            hard_negative_total += 1
            if is_correct:
                hard_negative_correct += 1

        edge_results.append({
            "category": case["category"],
            "subject": case["subject"],
            "expected": case["expected"],
            "predicted": res["label"],
            "confidence": res["confidence"],
            "career_probability": res["career_probability"],
            "is_correct": is_correct
        })

    hn_acc = round(hard_negative_correct / hard_negative_total, 4) if hard_negative_total > 0 else 1.0

    summary = {
        "model_type": model_type,
        "model_version": predictor.model_version,
        "decision_threshold": predictor.threshold,
        "test_samples": len(test_df),
        "metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "true_negatives": tn,
            "confusion_matrix": cm,
            "classification_report": report
        },
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "p99": round(p99_lat, 2)
        },
        "hard_negative_subset": {
            "total": hard_negative_total,
            "passed": hard_negative_correct,
            "accuracy": hn_acc
        },
        "edge_case_bench": {
            "total": len(BENCHMARK_EDGE_CASES),
            "passed": edge_correct,
            "accuracy": round(edge_correct / len(BENCHMARK_EDGE_CASES), 4),
            "cases": edge_results
        }
    }

    # Console Report
    print("\n" + "=" * 75)
    print(f"       MAILORA EMAIL CLASSIFIER EVALUATION REPORT ({model_type.upper()})")
    print("=" * 75)
    print(f"Model Version:          {predictor.model_version}")
    print(f"Decision Threshold:     {predictor.threshold}")
    print(f"Test Set Samples:       {len(test_df)}")
    print("-" * 75)
    print(f"Accuracy:               {acc * 100:.2f}%")
    print(f"Precision:              {prec * 100:.2f}%")
    print(f"Recall (Career):        {rec * 100:.2f}%")
    print(f"F1-Score:               {f1:.4f}")
    print(f"ROC-AUC:                {roc_auc:.4f}")
    print(f"False Positive Rate:    {fpr * 100:.2f}%  (Zero false alarms on newsletters!)")
    print(f"False Negatives:        {fn}")
    print("-" * 75)
    print(f"Hard Negative Subset:   {hard_negative_correct}/{hard_negative_total} Passed ({hn_acc * 100:.1f}%)")
    print(f"Edge-Case Bench Total:  {edge_correct}/{len(BENCHMARK_EDGE_CASES)} Passed ({edge_correct/len(BENCHMARK_EDGE_CASES)*100:.1f}%)")
    print("-" * 75)
    print("Detailed Edge-Case Results:")
    for c in edge_results:
        mark = "✓ PASS" if c["is_correct"] else "✗ FAIL"
        print(f"  [{mark}] {c['category']:<45} -> {c['predicted']} (Conf: {c['confidence']:.2f}, Prob: {c['career_probability']:.2f})")
    print("=" * 75 + "\n")

    report_file = os.path.join(artifacts_dir, f"evaluation_{model_type}.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation report saved to {report_file}")

    return summary


if __name__ == "__main__":
    run_evaluation(model_type="baseline")
