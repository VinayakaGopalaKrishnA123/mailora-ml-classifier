"""
Regression and Hard-Negative Test Suite for Mailora Email Classifier.

Verifies that the model correctly distinguishes actionable employment opportunities
(CAREER_RELATED) from educational newsletters, career advice, interview tutorials,
salary benchmarks, and tech market commentary (NON_CAREER_RELATED).
"""

import pytest
from src.predict import EmailPredictor


@pytest.fixture(scope="module")
def baseline_predictor():
    return EmailPredictor(model_type="baseline", artifacts_dir="artifacts")


# ─────────────────────────────────────────────────────────────────────────────
# Core Required Regression Tests (Section 8: Tests A through H)
# ─────────────────────────────────────────────────────────────────────────────

def test_regression_case_a_glasp_newsletter(baseline_predictor):
    """
    Test A:
    Subject: 'Why AI hasn't replaced software engineers, and won't'
    Content: General educational newsletter discussing AI, software engineering,
             labor-market research, and recommendations.
    Expected: NON_CAREER_RELATED
    """
    subject = "Why AI hasn't replaced software engineers, and won't"
    body = (
        "In this week's Glasp newsletter: We curate the best thinking on artificial intelligence, "
        "software engineering careers, and developer productivity. Featured Article: 'Why AI hasn't replaced "
        "software engineers, and won't' by Kevin Baker. Highlights from the piece: LLMs generate boilerplate code, "
        "but software engineering is fundamentally about understanding ambiguous business requirements, maintaining legacy systems, "
        "and architecture design. Plus: 5 articles on growth mindset, community highlights from fellow curators, "
        "and research publications on generative code models. Read the full reading list at https://glasp.co/newsletter-92"
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "NON_CAREER_RELATED"
    assert result["is_career_related"] is False
    assert result["career_probability"] < 0.50


def test_regression_case_b_internship_apply_now(baseline_predictor):
    """
    Test B:
    Subject: 'Software Engineer Internship — Apply Now'
    Content: A company announces an internship and provides application instructions.
    Expected: CAREER_RELATED
    """
    subject = "Software Engineer Internship — Apply Now"
    body = (
        "Applications are officially open for our Summer 2027 Software Engineering Internship at Datadog. "
        "Work on high-scale distributed monitoring and time-series engines. Open to pre-final year students graduating in 2028. "
        "Submit your resume, GitHub profile, and transcripts at https://careers.datadoghq.com/intern-swe before November 20."
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "CAREER_RELATED"
    assert result["is_career_related"] is True
    assert result["career_probability"] >= baseline_predictor.threshold


def test_regression_case_c_interview_tutorial(baseline_predictor):
    """
    Test C:
    Subject: 'How to crack coding interviews'
    Content: A tutorial about interview preparation.
    Expected: NON_CAREER_RELATED
    """
    subject = "How to crack coding interviews"
    body = (
        "Preparing for software engineering interviews at FAANG or top startups? In this detailed tutorial, "
        "we cover the most common dynamic programming patterns, binary tree traversals, and system design tradeoffs. "
        "Learn how to structure your behavioral answers using the STAR technique. Read our free interview preparation guide."
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "NON_CAREER_RELATED"
    assert result["is_career_related"] is False


def test_regression_case_d_interview_scheduled(baseline_predictor):
    """
    Test D:
    Subject: 'Your interview is scheduled for tomorrow'
    Content: A recruiter confirms an interview for a previously applied software engineering role.
    Expected: CAREER_RELATED
    """
    subject = "Your interview is scheduled for tomorrow"
    body = (
        "Hi Candidate, Your technical interview round for the Software Engineer position at Uber is confirmed "
        "for tomorrow, Thursday at 3:00 PM IST via Google Meet: https://meet.google.com/xyz-uber-interview. "
        "The interviewer will be a Senior Distributed Systems Engineer. Please have your CoderPad setup ready."
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "CAREER_RELATED"
    assert result["is_career_related"] is True


def test_regression_case_e_hiring_trends_newsletter(baseline_predictor):
    """
    Test E:
    Subject: 'Hiring trends in the technology industry'
    Content: An industry research newsletter with hiring statistics.
    Expected: NON_CAREER_RELATED
    """
    subject = "Hiring trends in the technology industry"
    body = (
        "According to our quarterly labor market research report, tech hiring has shifted toward specialized backend, "
        "cloud infrastructure, and AI engineering roles. Generalist hiring has contracted by 12% across Fortune 500 enterprises. "
        "Download the full employment research whitepaper on our economics research site."
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "NON_CAREER_RELATED"
    assert result["is_career_related"] is False


def test_regression_case_f_application_received(baseline_predictor):
    """
    Test F:
    Subject: 'Application received — Graduate Engineer Trainee'
    Content: Confirmation that an application for a specific position was received.
    Expected: CAREER_RELATED
    """
    subject = "Application received — Graduate Engineer Trainee"
    body = (
        "Dear Candidate, We acknowledge receipt of your application for the Graduate Engineer Trainee (GET) role "
        "at Tata Consultancy Services (Ref: TCS-GET-2026-902). Your resume and credentials are currently being reviewed "
        "by our talent acquisition panel. We will notify you regarding assessment dates shortly."
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "CAREER_RELATED"
    assert result["is_career_related"] is True


def test_regression_case_g_top_companies_editorial(baseline_predictor):
    """
    Test G:
    Subject: 'Top companies for software engineers'
    Content: A general editorial article ranking technology companies.
    Expected: NON_CAREER_RELATED
    """
    subject = "Top companies for software engineers"
    body = (
        "Our editorial ranking of the best technology workplaces for developers in 2026. We evaluated companies on "
        "developer culture, code review standards, remote work flexibility, and career advancement tracks. "
        "See where Stripe, Microsoft, and Datadog rank on this year's index."
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "NON_CAREER_RELATED"
    assert result["is_career_related"] is False


def test_regression_case_h_placement_drive_open(baseline_predictor):
    """
    Test H:
    Subject: 'Placement Drive Registration Open'
    Content: An official placement-cell email announcing a recruitment drive with eligible candidates,
             a company, and registration instructions.
    Expected: CAREER_RELATED
    """
    subject = "Placement Drive Registration Open: J.P. Morgan Campus Hiring 2027"
    body = (
        "Training & Placement Cell Announcement: J.P. Morgan Chase & Co. is visiting our campus for the 2027 batch recruitment drive. "
        "Role: Software Engineer Analyst. CTC: 19.5 LPA. Eligible Branches: B.Tech (CSE, IT, ECE) with CGPA >= 7.5 and no active backlogs. "
        "Eligible students must register on the campus placement portal before midnight tonight. Assessment date: Saturday."
    )
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "CAREER_RELATED"
    assert result["is_career_related"] is True


# ─────────────────────────────────────────────────────────────────────────────
# Additional Hard Negative Subsets (Section 9 Requirement)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("subject,body", [
    (
        "Top 50 LeetCode Questions for Software Engineering Technical Rounds",
        "Here is our curated collection of the top 50 algorithmic problems asked in technical screening interviews at top software companies."
    ),
    (
        "5 Common Mistakes on Software Engineer Resumes That Lead to Instant Rejection",
        "Recruiters review hundreds of tech applications every single day. Here are five resume mistakes software engineers make."
    ),
    (
        "Tech Salary Report 2026: What Software Engineers and Data Scientists Earn",
        "Levels.fyi and TechCompensation report: Comprehensive salary data for software engineers across global tech hubs."
    ),
    (
        "Annual Campus Placement Statistics Report 2025-2026",
        "The Dean of Student Affairs has published the annual college placement report. 94% of graduating engineering students were placed."
    ),
    (
        "Master System Design & Backend Architecture - Free Educational Webinar",
        "Join our weekend live masterclass on designing distributed caching and database sharding. Register for the educational session."
    ),
    (
        "Naukri Weekly Digest: Thousands of tech companies are looking for talent",
        "Keep your profile fresh and updated to catch the attention of recruiters. Browse trending categories on Naukri."
    ),
    (
        "The Senior Engineer Mindset: Moving beyond just writing code",
        "Writing code is only 30% of what distinguishes a senior software engineer. Communication and trade-off analysis matter much more."
    )
])
def test_hard_negative_evaluation_subset(baseline_predictor, subject, body):
    result = baseline_predictor.predict_single(subject, body)
    assert result["label"] == "NON_CAREER_RELATED", (
        f"Expected NON_CAREER_RELATED for '{subject}', but got {result['label']} (Prob: {result['career_probability']})"
    )
