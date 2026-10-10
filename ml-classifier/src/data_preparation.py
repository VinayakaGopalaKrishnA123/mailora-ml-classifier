"""
Dataset Preparation and Validation Pipeline for Mailora Email Classifier.

Generates, cleans, deduplicates, and stratifies realistic career and non-career
email datasets, including a comprehensive Hard Negative benchmark (educational
newsletters, interview prep tutorials, tech industry trends, and salary reports)
to prevent false positives on career-adjacent keywords.
"""

import os
import re
import random
import logging
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import clean_text, text_dedup_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fixed seed for complete reproducibility
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# 1. Career-Related Seed Samples (Actionable Opportunities)
# ─────────────────────────────────────────────────────────────────────────────

def get_career_seed_samples() -> List[Dict[str, str]]:
    """
    Returns realistic career-related email samples spanning actionable opportunities:
    - Job vacancies with direct application instructions
    - Internship openings & research cohorts
    - Campus placement drives with eligibility criteria
    - Online coding assessments & hackathons with hiring tracks
    - Technical interview scheduling & meet links
    - Application status updates (under review, shortlisted, rejected)
    - Formal offer letters, compensation breakdowns, and onboarding forms
    - Graduate engineer trainees (GET) and fresher hiring alerts
    - Job application deadline reminders
    """
    samples = [
        # Job Openings & Direct Hiring
        {
            "subject": "Job Opening: Backend Software Engineer (Node.js/Python) at Razorpay",
            "body": "Hi there, Razorpay is hiring experienced Backend Software Engineers for our Core Banking platform in Bengaluru. Responsibilities include building scalable distributed microservices, optimizing database transactions, and designing fault-tolerant payment flows. Requirements: 2+ years of experience with Node.js, Go, or Python, Kafka, and PostgreSQL. Apply now at https://razorpay.com/careers/backend-eng-2026 before October 30.",
        },
        {
            "subject": "LinkedIn Job Alert: 5 new Senior Frontend Engineer roles matching your preferences",
            "body": "Your daily job alert is ready. Roles matching your search 'Frontend Engineer (React/TypeScript)': 1. Senior UI Engineer at Atlassian (Remote) - Apply via Workday. 2. Frontend Architect at Stripe (Bengaluru). 3. Staff Engineer at Swiggy. Click here to view the full job descriptions and submit your application: https://linkedin.com/jobs/view/987123",
        },
        {
            "subject": "We are hiring: Cloud DevOps Engineer at Cisco Systems",
            "body": "Dear Candidate, We came across your profile and believe your cloud infrastructure background would be a great fit for our Cloud Security Engineering team at Cisco. You will be responsible for Kubernetes orchestration, Terraform automation, and CI/CD pipelines. Please submit your updated resume and portfolio through our careers portal: https://jobs.cisco.com/req/194821",
        },
        {
            "subject": "Exciting Career Opportunity: Data Scientist - NLP / LLMs at Microsoft",
            "body": "Microsoft is seeking a talented Data Scientist to work with the Azure AI Cognitive Services group. You will research, fine-tune, and deploy transformer models and multi-modal agents for enterprise customers. Qualifications: M.S. or B.S. in Computer Science, solid understanding of PyTorch and NLP. Review the job details and apply today.",
        },
        {
            "subject": "Hiring Alert: Full Stack Developer (React + Spring Boot) at Infosys BPM",
            "body": "Greetings! Infosys is hiring Full Stack Engineers across Hyderabad, Pune, and Chennai locations. Candidates with 1-4 years of hands-on experience in Java Spring Boot, microservices architecture, React.js, and SQL are invited to apply. Immediate joiners preferred. Submit your application and CTC details to recruitment@infosys.com.",
        },
        {
            "subject": "Software Engineer Internship — Apply Now",
            "body": "Applications are officially open for our Summer 2027 Software Engineering Internship at Datadog. Work on high-scale distributed monitoring and time-series engines. Open to pre-final year students graduating in 2028. Submit your resume, GitHub profile, and transcripts at https://careers.datadoghq.com/intern-swe before November 20.",
        },

        # Internship Opportunities
        {
            "subject": "Summer 2027 Software Engineering Internship - Applications Now Open!",
            "body": "Google is excited to announce that applications for the Summer 2027 Software Engineer Summer Internship program are officially open. Open to undergraduate and master's students graduating between December 2027 and June 2028. Work alongside world-class engineers on real-world projects. Application deadline is November 15, 2026. Submit your resume and transcripts at https://careers.google.com/internships",
        },
        {
            "subject": "Research Internship Opportunity: Machine Learning Lab at IIT Bombay",
            "body": "Applications are invited for a 6-month funded research internship in Deep Learning for Computer Vision and Medical Imaging at IIT Bombay. Selected interns will receive a monthly stipend of INR 25,000 and working directly on peer-reviewed publications. Apply with your CV, GitHub link, and statement of purpose by October 25th.",
        },
        {
            "subject": "Adobe Product Design & Frontend Internship - Winter Cohort",
            "body": "Adobe is looking for curious student developers for our 3-month winter internship program. You will build user experiences in Creative Cloud using React, WebAssembly, and modern web standards. Stipend: INR 60,000/month. Eligible: Pre-final year B.Tech students. Apply on the university recruiting portal.",
        },
        {
            "subject": "Internship Alert: Data Analyst Intern at Swiggy",
            "body": "Swiggy Analytics is hiring Data Analyst interns. Work with business intelligence teams to build dashboards, analyze consumer retention, and write high-performance SQL queries. Duration: 6 months (Full-time). Stipend provided. To apply, fill out the Google Form before 5 PM tomorrow.",
        },

        # Campus Placements & Recruitment Drives
        {
            "subject": "Placement Drive Registration Open: J.P. Morgan Campus Hiring 2027",
            "body": "Dear Final Year Students, Training & Placement Cell announces the on-campus recruitment drive for J.P. Morgan Chase & Co. Role: Software Engineer Analyst. CTC: 19.5 LPA. Eligibility: B.Tech (CSE/IT/ECE) with CGPA >= 7.5 and no active backlogs. Interested eligible candidates must register on the Superset portal by 23:59 hrs on October 14. Online assessment will be held on October 18.",
        },
        {
            "subject": "T&P Cell Announcement: TCS National Qualifier Test (NQT) Registration Deadline",
            "body": "Notice to all 2027 graduating batch students: Registration for TCS NQT 2026-2027 is closing on Friday. Roles offered: TCS Ninja and TCS Digital (7 LPA - 9 LPA). All eligible students must register on the TCS NextStep portal and submit their registration numbers to the placement coordinator.",
        },
        {
            "subject": "Campus Drive: Deloitte USI Risk & Financial Advisory - Shortlisted Candidates",
            "body": "Placement Cell Update: Please find attached the list of candidates shortlisted for the technical and HR interviews with Deloitte USI scheduled for tomorrow at 9:00 AM in the Main Seminar Hall. Please bring two copies of your resume, college ID card, and formal dress code is mandatory.",
        },
        {
            "subject": "Graduate Engineer Trainee Recruitment — Apply Now",
            "body": "L&T Technology Services is inviting applications for the Graduate Engineer Trainee (GET) Batch of 2026. Opportunities in Embedded Systems, Industrial IoT, and Automotive Software. Training period: 12 months with full absorption into engineering units upon completion. Apply on the LTTS campus portal.",
        },

        # Online Assessments & Coding Challenges
        {
            "subject": "Amazon Online Assessment: Software Development Engineer I - Action Required",
            "body": "Hello Candidate, Thank you for applying for the SDE I role at Amazon. You are invited to take the Amazon Online Assessment on HackerRank. The assessment consists of two coding challenges (Data Structures and Algorithms) and a work style simulation. Duration: 90 minutes. You have 5 days to complete this assessment. Click here to start: https://amazon.hackerrank.com/test/sde1-invite-921",
        },
        {
            "subject": "Coding Test Invitation: Goldman Sachs Engineering Campus Hiring",
            "body": "Dear Applicant, Your profile has been screened for the Goldman Sachs 2026-2027 hiring assessment. Please log in to the HackerRank assessment platform using your registered email. Test window: Saturday, October 17 from 10:00 AM to 12:00 PM IST. Webcam proctoring is enabled. Best of luck!",
        },
        {
            "subject": "Uber CodeSignal General Coding Assessment Invitation",
            "body": "You have received an invitation from Uber Technologies to complete the CodeSignal General Coding Assessment (GCA) for the Software Engineer New Grad position. You will have 70 minutes to solve 4 algorithmic tasks. Please ensure you are in a quiet room with a working camera.",
        },

        # Interview Invitations & Confirmations
        {
            "subject": "Your interview is scheduled for tomorrow: Software Engineer Intern at Uber",
            "body": "Hi Candidate, Your technical interview round for the Software Engineer Intern position at Uber is confirmed for tomorrow, Thursday at 3:00 PM IST via Google Meet: https://meet.google.com/xyz-uber-interview. The interviewer will be a Senior Distributed Systems Engineer. Please have your CoderPad setup ready.",
        },
        {
            "subject": "Interview Invitation: Technical Round 1 - Backend Engineer at Cred",
            "body": "Hi Candidate, Congratulations! Following your coding assessment, we would like to invite you for Technical Interview Round 1 with our engineering manager. The interview will focus on system design, database scaling, and low-level architecture. Date: Thursday, Oct 15, 2026 at 3:00 PM IST via Google Meet: https://meet.google.com/xyz-abcd-efg. Please confirm your availability.",
        },
        {
            "subject": "Interview Schedule: Google Meet link for Stripe Virtual Onsite - Round 2",
            "body": "Dear Candidate, We have scheduled your second technical interview for the Infrastructure Engineer role at Stripe. Interviewer: Alex Johnson (Staff Infrastructure Engineer). Topic: Distributed Systems & Concurrency. Time: October 19, 2026, 11:00 AM - 12:00 PM IST. Video link: https://meet.google.com/str-infra-onsite",
        },

        # Application Status (Received, Shortlisted, Rejection)
        {
            "subject": "Application received — Graduate Engineer Trainee at Tata Consultancy Services",
            "body": "Dear Candidate, We acknowledge receipt of your application for the Graduate Engineer Trainee (GET) role at TCS (Ref: TCS-GET-2026-902). Your resume and credentials are currently being reviewed by our talent acquisition panel. We will notify you regarding assessment dates shortly.",
        },
        {
            "subject": "Congratulations! You have been shortlisted for Microsoft Aspire Program",
            "body": "Dear Applicant, We are excited to inform you that your profile has been shortlisted for the next phase of the Microsoft Aspire University Hiring program. The next stage involves a technical coding test followed by virtual technical interviews. Please watch your inbox for the scheduling link.",
        },
        {
            "subject": "Update on your job application for Software Engineer at Stripe",
            "body": "Thank you for taking the time to interview with Stripe for the Software Engineer role. While our interviewers enjoyed learning about your background and technical projects, we have decided to move forward with another candidate whose experience more closely matches our immediate requirements. We encourage you to apply for future opportunities on our careers page.",
        },
        {
            "subject": "Your application status for Associate Software Engineer at Morgan Stanley",
            "body": "Dear Candidate, Thank you for your interest in Morgan Stanley and for completing our technical assessment. After careful review, we regret to inform you that we will not be moving forward with your application for the Associate Software Engineer position at this time. We wish you the best in your job search.",
        },

        # Offer Letters & Onboarding
        {
            "subject": "Offer of Employment: Software Development Engineer at Oracle",
            "body": "Dear Candidate, On behalf of Oracle India Private Limited, we are thrilled to extend an offer of employment for the position of Software Development Engineer (IC2). Your annual base salary, signing bonus, and equity RSUs are detailed in the attached offer letter. Please review, sign, and return the duplicate copy by October 25 to initiate pre-boarding.",
        },
        {
            "subject": "Congratulations! Offer Letter from Salesforce - Welcome to the Ohana",
            "body": "We are delighted to offer you the position of Member of Technical Staff (MTS) at Salesforce! Attached please find your formal Employment Agreement, CTC structure, and benefits summary. Please sign through DocuSign and submit your background verification documents to begin onboarding.",
        },
        {
            "subject": "Reminder: Deadline Approaching for Goldman Sachs Summer Analyst Internship",
            "body": "This is a friendly reminder that the deadline to submit your application for the Goldman Sachs 2027 Summer Analyst Program is October 20 at 11:59 PM. Incomplete applications will not be evaluated for coding test invitations. Submit your application now at https://goldmansachs.com/careers",
        },
    ]
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 2. Hard Negative Seed Samples (Tech / Career Discussion without Actionable Jobs)
# ─────────────────────────────────────────────────────────────────────────────

def get_hard_negative_seed_samples() -> List[Dict[str, str]]:
    """
    Returns realistic negative samples (label 0) that contain career, tech, AI,
    software engineering, and hiring terminology, but represent educational content,
    industry analysis, tutorials, or newsletters rather than actionable jobs.
    """
    samples = [
        # 1. Exact Glasp Newsletter (User Specified Case)
        {
            "subject": "Why AI hasn't replaced software engineers, and won't",
            "body": (
                "In this week's Glasp newsletter: We curate the best thinking on artificial intelligence, "
                "software engineering careers, and developer productivity.\n\n"
                "Featured Article: 'Why AI hasn't replaced software engineers, and won't' by Kevin Baker.\n"
                "Highlights from the piece: LLMs generate boilerplate code efficiently, but modern software engineering "
                "is fundamentally about understanding ambiguous business requirements, maintaining legacy systems, and architecture design. "
                "Software engineers who learn AI tooling will be more productive, not obsolete.\n\n"
                "Plus in this issue:\n"
                "- 5 articles on growth mindset and learning habit loops\n"
                "- Community highlights from fellow knowledge curators\n"
                "- Research publications on generative code models and benchmark evaluations\n"
                "- Upcoming community read-along on developer tools\n\n"
                "Check out the full reading list and save your highlights: https://glasp.co/newsletter-92"
            ),
        },

        # 2. Tech Newsletters & Industry Trends
        {
            "subject": "The Pragmatic Engineer: How software engineering careers are evolving in 2026",
            "body": (
                "Welcome to this week's edition of The Pragmatic Engineer. Today we look at how software engineering "
                "career ladders have adapted to AI tooling, autonomous agents, and full-stack ownership. "
                "We surveyed 1,200 engineering managers across Big Tech and venture-backed startups on hiring velocity, "
                "seniority expectations, and compensation bands. Read the full analysis on Substack."
            ),
        },
        {
            "subject": "ByteByteGo Newsletter: Software Architecture, AI Systems, and Engineering Career Growth",
            "body": (
                "In this issue: Microservices vs Monoliths in 2026. How top tech companies design fault-tolerant systems. "
                "We also share advice on navigating software engineering careers from junior developer to principal architect. "
                "Read the complete visual breakdown and subscribe to our YouTube channel."
            ),
        },

        # 3. Interview Preparation Guides & Tutorials (Educational, NOT an Interview Invite)
        {
            "subject": "How to crack coding interviews: A complete roadmap for SDE roles",
            "body": (
                "Preparing for software engineering interviews at FAANG or top startups? In this detailed tutorial, "
                "we cover the most common dynamic programming patterns, binary tree traversals, and system design tradeoffs. "
                "Learn how to structure your behavioral answers using the STAR technique. Read our free interview preparation guide."
            ),
        },
        {
            "subject": "Top 50 LeetCode Questions for Software Engineering Technical Rounds",
            "body": (
                "Here is our curated collection of the top 50 algorithmic problems asked in technical screening interviews "
                "at top software companies. Categorized by patterns: Two Pointers, Sliding Window, Monotonic Stack, and Graphs. "
                "Practice these questions to sharpen your problem-solving skills for upcoming interview seasons."
            ),
        },

        # 4. Career Advice, Resume-Writing, and Portfolio Tutorials
        {
            "subject": "5 Common Mistakes on Software Engineer Resumes That Lead to Instant Rejection",
            "body": (
                "Recruiters review hundreds of tech applications every single day. Here are five resume mistakes "
                "software engineers make: listing 35 tools without business impact, missing GitHub project links, "
                "and poor formatting. Follow our step-by-step resume optimization guide to get noticed by engineering hiring managers."
            ),
        },
        {
            "subject": "How to Optimize Your LinkedIn Profile as a Software Developer",
            "body": (
                "Learn how to position your technical experience, open-source contributions, and engineering portfolio "
                "on LinkedIn so tech recruiters find your profile. Tips on headline optimization, project descriptions, and networking."
            ),
        },

        # 5. Salary Trends, Compensation Benchmarks, and Labor Market Research
        {
            "subject": "Tech Salary Report 2026: What Software Engineers and Data Scientists Earn",
            "body": (
                "Levels.fyi and TechCompensation report: Comprehensive salary data for software engineers, product managers, "
                "and engineering leads. We break down base compensation, stock options (RSUs), and median signing bonuses "
                "across Bengaluru, London, and San Francisco. View the full interactive salary benchmark charts."
            ),
        },
        {
            "subject": "Hiring trends in the technology industry: What the data tells us",
            "body": (
                "According to our quarterly labor market research report, tech hiring has shifted toward specialized backend, "
                "cloud infrastructure, and AI engineering roles. Generalist hiring has contracted by 12% across Fortune 500 enterprises. "
                "Download the full employment research whitepaper."
            ),
        },

        # 6. General News / Technology Articles Mentioning Hiring in Passing
        {
            "subject": "Top companies for software engineers in 2026: Culture, Pay, and Growth",
            "body": (
                "Our editorial ranking of the best technology workplaces for developers. We evaluated companies on developer culture, "
                "code review standards, remote work flexibility, and career advancement tracks. See where Stripe, Microsoft, "
                "and Datadog rank on this year's index."
            ),
        },
        {
            "subject": "Tech Giant plans to hire 5,000 workers for new AI cloud campus",
            "body": (
                "News report: Amazon Web Services announced plans to invest $10 billion in regional cloud data centers, "
                "potentially creating over 5,000 engineering and operations roles over the next five years. Industry analysts "
                "comment on what this means for the broader tech ecosystem."
            ),
        },
        {
            "subject": "Layoffs and Restructuring Across Tech Startups: An Economic Analysis",
            "body": (
                "An investigative analysis on venture funding retrenchment and tech workforce adjustments. Why growth-stage "
                "startups are prioritizing profitability over aggressive hiring, and how engineers are adapting to longer search cycles."
            ),
        },

        # 7. University Placement Statistics / Overview (NOT an Active Drive Announcement)
        {
            "subject": "Annual Campus Placement Statistics Report 2025-2026",
            "body": (
                "The Dean of Student Affairs has published the annual college placement report. 94% of graduating engineering students "
                "were placed across 150 visiting companies with a median CTC of 11.8 LPA. The report highlights historical placement "
                "trends over the last five years. Download the statistical review from the college portal."
            ),
        },

        # 8. Course & Bootcamp Advertisements (Educational Marketing)
        {
            "subject": "Master System Design & Backend Architecture - Free Educational Webinar",
            "body": (
                "Want to learn how to design systems handling 10 million concurrent users? Join our weekend live masterclass "
                "led by former staff engineers from Meta. We explore caching strategies, database replication, and message queues. "
                "Register for the free educational session at https://scalingschool.dev/webinar"
            ),
        },
        {
            "subject": "Full Stack Development Certification with Placement Assistance - 50% Off",
            "body": (
                "Upgrade your tech stack with our 24-week Full Stack MERN course. Includes live doubt sessions, project reviews, "
                "and career counseling workshops. Note: This is an educational training bootcamp. Enroll today to claim your discount."
            ),
        },

        # 9. Job Portal Promotional Newsletters with No Specific Job
        {
            "subject": "Naukri Weekly Digest: Thousands of tech companies are looking for talent",
            "body": (
                "Keep your Naukri profile fresh and updated to catch the attention of recruiters. Update your primary technical skills, "
                "upload an updated resume, and explore trending career categories in Software Development, Cloud Computing, and DevOps."
            ),
        },
        {
            "subject": "Indeed Career Newsletter: Strategies for your job search journey",
            "body": (
                "In this week's career newsletter from Indeed: How to stay motivated during an extended job search, tips for building "
                "meaningful professional relationships on LinkedIn, and how to negotiate salary offers. Read the full article."
            ),
        },

        # 10. Developer Mindset, Productivity & Essays
        {
            "subject": "The Senior Engineer Mindset: Moving beyond just writing code",
            "body": (
                "Writing code is only 30% of what distinguishes a senior software engineer. Communication, cross-functional alignment, "
                "and trade-off analysis matter much more. In this essay, we break down frameworks for professional growth in tech."
            ),
        },
    ]
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 3. Everyday Non-Career Seed Samples (Shopping, Banking, Personal, Spam)
# ─────────────────────────────────────────────────────────────────────────────

def get_standard_non_career_seed_samples() -> List[Dict[str, str]]:
    """
    Standard everyday non-career emails: shopping, deliveries, banking, personal chats.
    """
    samples = [
        # Shopping & Orders
        {
            "subject": "Your Amazon.in order #402-9182741 has been dispatched!",
            "body": "Hello, your order containing 'Logitech MX Master 3S Wireless Mouse' has been dispatched via BlueDart. Expected delivery: Tomorrow by 8 PM. Track your parcel anytime on the Amazon app.",
        },
        {
            "subject": "Delivered: Your Flipkart order has reached your doorstep",
            "body": "Great news! Your package containing 'Sony WH-1000XM5 Noise Cancelling Headphones' has been delivered to your address. Rate your delivery experience on Flipkart.",
        },
        {
            "subject": "Your Swiggy order from Biryani Blues is arriving in 15 mins",
            "body": "Rider Ramesh is on the way with your food. Track order live on Swiggy app. Total paid: INR 350 via UPI.",
        },
        {
            "subject": "Zomato Receipt: Order confirmed from Starbucks Coffee",
            "body": "We have received your payment of INR 450 via UPI. Order details: 1x Caramel Frappuccino, 1x Croissant. Estimated preparation time: 10 mins. Track order on Zomato.",
        },

        # Banking & Finance
        {
            "subject": "HDFC Bank: Monthly e-Statement for Credit Card ending 4092",
            "body": "Dear Customer, Your credit card e-statement for the billing cycle ending October 2026 is attached. Total Amount Due: INR 14,820.50. Minimum Due: INR 1,200. Payment Due Date: 28-Oct-2026. Pay easily through NetBanking.",
        },
        {
            "subject": "ICICI Bank Alert: INR 2,500.00 debited from Account XX1092 via UPI",
            "body": "Dear Customer, Account XX1092 has been debited for INR 2,500.00 on 10-Oct-2026 14:22:10 towards UPI transaction ref 98124019284. If unauthorized, call 1800-ICICI immediately.",
        },
        {
            "subject": "State Bank of India: Your OTP for NetBanking Login is 492019",
            "body": "OTP 492019 is your One Time Password for SBI NetBanking login. Valid for 3 minutes. Do not share this OTP with anyone, including bank officials.",
        },
        {
            "subject": "Groww: Mutual Fund SIP transaction successful for October",
            "body": "Your monthly SIP installment of INR 5,000 for Parag Parikh Flexi Cap Fund has been successfully processed. Units will be allotted within 2 business days according to the prevailing NAV.",
        },

        # Personal Conversations
        {
            "subject": "Dinner this Saturday at 8 PM?",
            "body": "Hey, hope you're having a great week! A few of us are planning to meet up for dinner this Saturday around 8 PM at the Italian place downtown. Let me know if you can make it so I can reserve a table for us.",
        },
        {
            "subject": "Photos from our weekend hiking trip!",
            "body": "Hey everyone, I just uploaded all the photos and drone videos from our trip to the shared Google Drive folder. You can download and look through them here: https://drive.google.com/hiking-photos.",
        },
        {
            "subject": "Dinner plans tonight?",
            "body": "Hey, are you free for a quick coffee or dinner tonight around 7 PM? Let me know!",
        },
        {
            "subject": "Great job on the presentation today!",
            "body": "Hi team, you all did a fantastic job on the internal client pitch this morning. The slides and demo were crisp. Have a wonderful weekend!",
        },

        # Subscriptions, Travel & General News
        {
            "subject": "Netflix: New movies and series added for you this weekend",
            "body": "Stranger Things Season 5 is now streaming! Plus explore our top 10 trending thriller series, anime, and blockbuster movies. Grab your popcorn and start watching tonight on Netflix.",
        },
        {
            "subject": "Spotify Premium: Enjoy 3 months for just INR 59",
            "body": "Listen to your favorite music ad-free, download songs for offline listening, and get unlimited skips. Special festive offer valid until October 31. Upgrade to Spotify Premium today.",
        },
        {
            "subject": "MakeMyTrip: Flight Ticket Confirmation - Bengaluru to Delhi (PNR: 98124X)",
            "body": "Your Indigo flight 6E-204 from BLR to DEL is confirmed for 24-Oct-2026 at 07:15 AM. Web check-in opens 48 hours prior to departure. Terminal: T2. View boarding pass on MakeMyTrip.",
        },
        {
            "subject": "AWS Cloud Security Alert: Root account login detected",
            "body": "Amazon Web Services detected a root login to your AWS account (ID: 98124192) from IP 49.37.12.1 at 2026-10-10 11:20 UTC. If this was not you, rotate your credentials immediately.",
        },
        {
            "subject": "The Morning Brew: Tech earnings, oil prices, and the latest market moves",
            "body": "Good morning! Here is what's driving markets today: Nasdaq climbs 1.2% led by semiconductor earnings, European central banks consider interest rate cuts. Grab your coffee and read our 5-minute digest.",
        },
    ]
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. Synthetic Variation Generators (Balanced & Adversarial Hard Negatives)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_variations(target_label: int, count: int = 180) -> List[Dict[str, any]]:
    """
    Generates balanced, high-fidelity variations.
    For label 0 (Non-Career), includes standard everyday negatives AND adversarial
    hard negatives (tech newsletters, interview prep blogs, salary surveys, hiring news).
    For label 1 (Career), includes actionable recruitment communications with clear CTAs.
    """
    random.seed(RANDOM_SEED)

    tech_companies = [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "NVIDIA", "Netflix",
        "Atlassian", "Uber", "Stripe", "Goldman Sachs", "Morgan Stanley", "J.P. Morgan",
        "Flipkart", "Razorpay", "Swiggy", "Zomato", "Cred", "PhonePe", "Postman",
        "Infosys", "TCS", "Wipro", "Cognizant", "Accenture", "Cisco", "Oracle",
        "Adobe", "Salesforce", "Databricks", "Snowflake", "Palantir", "Bloomberg"
    ]

    tech_roles = [
        "Software Engineer", "Frontend Developer", "Backend Engineer",
        "Full Stack Developer", "Machine Learning Engineer", "Data Scientist",
        "DevOps Engineer", "Cloud Solutions Architect", "QA Automation Engineer",
        "Data Analyst", "Systems Engineer", "Graduate Engineer Trainee"
    ]

    intern_roles = [
        "Software Engineering Intern", "Machine Learning Research Intern",
        "Data Science Summer Intern", "Frontend Development Intern",
        "Cloud Infrastructure Intern"
    ]

    locations = ["Bengaluru", "Hyderabad", "Pune", "Gurugram", "Noida", "Mumbai", "Chennai", "Remote (India)"]
    coding_platforms = ["HackerRank", "LeetCode Contest", "CodeSignal", "Mettl"]

    dataset = []

    if target_label == 1:
        # ACTIONABLE CAREER SAMPLES (label = 1)
        for i in range(count):
            comp = random.choice(tech_companies)
            role = random.choice(tech_roles)
            loc = random.choice(locations)
            plat = random.choice(coding_platforms)
            template_type = random.choice([
                "job_opening", "internship", "coding_test", "interview_invite",
                "application_shortlist", "rejection", "offer_letter", "campus_drive", "deadline_reminder"
            ])

            if template_type == "job_opening":
                subject = f"{comp} is hiring: {role} in {loc} — Apply Now"
                body = (
                    f"Our engineering team at {comp} ({loc}) is actively recruiting for a {role}. "
                    f"Responsibilities include building high-throughput microservices, optimizing database performance, "
                    f"and designing fault-tolerant distributed systems. "
                    f"Qualifications: 1-4 years of experience with Python, Go, or Java. "
                    f"Submit your resume and portfolio at https://careers.{comp.lower().replace(' ', '')}.com/jobs/{1000+i} before the deadline."
                )
            elif template_type == "internship":
                irole = random.choice(intern_roles)
                subject = f"{comp} Summer 2027 Internship Applications: {irole}"
                body = (
                    f"Applications are now open for the Summer 2027 {irole} cohort at {comp} ({loc}). "
                    f"Selected interns will work alongside core engineering teams on production infrastructure. "
                    f"Open to pre-final year engineering students. Stipend and housing allowance provided. "
                    f"Apply today on our careers portal: https://careers.{comp.lower().replace(' ', '')}.com/interns"
                )
            elif template_type == "coding_test":
                subject = f"Invitation: {comp} Technical Coding Assessment on {plat} for {role}"
                body = (
                    f"Dear Candidate, Thank you for applying for the {role} position at {comp}. "
                    f"You have been invited to complete our online technical assessment on {plat}. "
                    f"The test consists of algorithmic coding challenges and system problem-solving (90 minutes). "
                    f"Please complete your assessment within 4 days: https://{plat.lower().replace(' ', '')}.com/test/invite-{5000+i}"
                )
            elif template_type == "interview_invite":
                round_name = random.choice(["Round 1 (Technical Screening)", "Round 2 (System Design)", "Round 3 (Hiring Manager)"])
                subject = f"Interview Schedule: {comp} {round_name} - {role}"
                body = (
                    f"Hi Candidate, Following your performance on the technical assessment for {role} at {comp}, "
                    f"we would like to schedule your {round_name}. The 60-minute session will take place via Google Meet / Zoom. "
                    f"Please confirm your availability for tomorrow or choose an available slot on our interview scheduler."
                )
            elif template_type == "application_shortlist":
                subject = f"Your application for {role} at {comp} has been Shortlisted"
                body = (
                    f"Congratulations! Your profile has been screened and shortlisted by the engineering hiring committee at {comp}. "
                    f"Our technical recruiter will contact you shortly to coordinate your virtual onsite interviews. "
                    f"Please keep your resume and GitHub links up to date."
                )
            elif template_type == "rejection":
                subject = f"Update regarding your job application for {role} at {comp}"
                body = (
                    f"Dear Candidate, Thank you for taking the time to interview for the {role} position at {comp}. "
                    f"While our interviewers appreciated your experience and technical discussion, we have decided not to move "
                    f"forward with your candidacy for this role at this time. We will keep your resume on file for future opportunities."
                )
            elif template_type == "offer_letter":
                subject = f"Formal Offer of Employment: {role} at {comp}"
                body = (
                    f"Dear Candidate, On behalf of {comp}, we are thrilled to extend a formal offer of employment for the role of {role}. "
                    f"Your annual compensation breakdown, signing bonus, and joining benefits are outlined in the attached letter. "
                    f"Please review, sign, and return the duplicate copy within 5 business days to begin pre-boarding formalities."
                )
            elif template_type == "campus_drive":
                subject = f"Placement Drive Registration Open: {comp} On-Campus Hiring for {role}"
                body = (
                    f"Training & Placement Cell Announcement: {comp} is visiting campus for the 2027 batch recruitment drive. "
                    f"Position: {role}. Eligible Branches: B.Tech (CSE, IT, ECE) with CGPA >= 7.0 and no active backlogs. "
                    f"Eligible students must register on the campus placement portal before midnight tonight. Assessment date: Saturday."
                )
            else: # deadline_reminder
                subject = f"Deadline Reminder: Apply for {comp} {role} before portal closes"
                body = (
                    f"This is an urgent reminder that applications for the {role} opening at {comp} will close in 24 hours. "
                    f"Ensure your application, resume, and questionnaire are submitted through our careers portal before the deadline."
                )

            dataset.append({"subject": subject, "body": body, "label": 1})

    else:
        # NON-CAREER SAMPLES (label = 0)
        # Includes BOTH standard everyday negatives AND adversarial hard negatives (newsletters, tutorials, trends, guides)
        non_career_brands = ["Amazon", "Flipkart", "Swiggy", "Zomato", "Uber", "Netflix", "Spotify"]
        non_career_banks = ["HDFC Bank", "ICICI Bank", "State Bank of India", "Axis Bank"]
        personal_names = ["Rahul", "Priya", "Amit", "Sneha", "Karan", "Ananya"]

        for i in range(count):
            # Mix 50% Hard Negatives (containing tech/career keywords) and 50% Standard Negatives (banking, shopping, personal)
            is_hard_negative = (i % 2 == 0)

            if is_hard_negative:
                # ADVERSARIAL HARD NEGATIVE (Contains "software engineer", "AI", "interview", "hiring", but NOT a job!)
                hard_type = random.choice([
                    "ai_essay", "career_trends_newsletter", "interview_tutorial",
                    "salary_report", "placement_stats_report", "news_report_hiring",
                    "job_search_advice", "tech_rankings", "dev_mindset"
                ])

                if hard_type == "ai_essay":
                    subject = f"Why AI will not replace {random.choice(['software engineers', 'junior developers', 'data scientists'])}"
                    body = (
                        f"In this week's technology analysis newsletter: Why generative models and coding agents augment rather than replace "
                        f"software engineers. We break down the cognitive tasks in software architecture: context gathering, legacy code maintenance, "
                        f"and cross-functional alignment. Read the full essay and discussion on our developer blog."
                    )
                elif hard_type == "career_trends_newsletter":
                    subject = f"Software Engineering Career Digest: Trends, AI Tools, and Industry News"
                    body = (
                        f"This week's engineering leadership newsletter covers shifting expectations for software engineers in 2026. "
                        f"How engineering managers evaluate technical breadth, code reviews, and system ownership. "
                        f"Includes 5 recommended reads, an interview with a tech founder, and research publications on developer velocity."
                    )
                elif hard_type == "interview_tutorial":
                    subject = f"How to prepare for coding and system design interviews: Complete Tutorial"
                    body = (
                        f"Preparing for technical rounds at top tech companies? In this educational tutorial, we break down top algorithmic "
                        f"patterns for software engineers: dynamic programming, graphs, and LRU caches. Learn how to ace your next technical discussion. "
                        f"Bookmark this free guide on our programming tutorials site."
                    )
                elif hard_type == "salary_report":
                    subject = f"2026 Tech Salary Report: Compensation trends for software engineers"
                    body = (
                        f"Our annual compensation benchmark report is now published. We surveyed 15,000 software engineers, backend leads, "
                        f"and DevOps engineers across global tech hubs on median base salaries, bonuses, and equity compensation trends. "
                        f"Explore the interactive salary charts on our research portal."
                    )
                elif hard_type == "placement_stats_report":
                    subject = f"University Placement Cell Annual Statistics Report: 2025-2026 Review"
                    body = (
                        f"The Dean of Academic Affairs shares the annual placement statistics report for our graduating batch. "
                        f"Overall placement rate reached 93% across engineering departments with strong participation from 120 recruiters. "
                        f"This document provides statistical data and historical trends for research and accreditation purposes."
                    )
                elif hard_type == "news_report_hiring":
                    comp = random.choice(tech_companies)
                    subject = f"{comp} plans to expand technical workforce with 1,000 new roles in Europe"
                    body = (
                        f"According to business press reports, {comp} announced plans to expand its research and development operations in Europe, "
                        f"investing in cloud infrastructure and hiring engineers over the next two years. Financial analysts discuss market impact."
                    )
                elif hard_type == "job_search_advice":
                    subject = f"10 Job-Search and Resume Tips for Aspiring Software Developers"
                    body = (
                        f"Navigating the tech job market can be challenging. In this career advice article, we share practical strategies "
                        f"for crafting impactful resume bullet points, building open-source projects, and networking with engineering managers on LinkedIn."
                    )
                elif hard_type == "tech_rankings":
                    subject = f"Top 20 Technology Companies for Software Engineers in 2026"
                    body = (
                        f"Our annual workplace ranking evaluates the top technology employers for software engineers based on engineering culture, "
                        f"mentorship, developer productivity tooling, and work-life balance. Check out the full ranked list on our editorial portal."
                    )
                else: # dev_mindset
                    subject = f"The Productive Developer: Habits of Effective Software Engineers"
                    body = (
                        f"What separates great software engineers from average coders? In this reflective newsletter article, we explore deep work habits, "
                        f"managing code review feedback, and building lasting engineering careers. Read more on our developer community site."
                    )

            else:
                # STANDARD EVERYDAY NEGATIVE (Shopping, Banking, Personal, Travel)
                std_type = random.choice(["shopping", "banking", "personal", "travel", "subscription"])
                brand = random.choice(non_career_brands)
                bank = random.choice(non_career_banks)
                name = random.choice(personal_names)

                if std_type == "shopping":
                    subject = f"Your {brand} order #{random.randint(100000, 999999)} has been confirmed"
                    body = (
                        f"Thank you for ordering with {brand}! Your package containing electronics has been packed and dispatched. "
                        f"Estimated delivery time: 2 business days. Track your order live on the {brand} mobile app."
                    )
                elif std_type == "banking":
                    amt = random.randint(200, 7500)
                    subject = f"{bank} Alert: Account debited for INR {amt}.00 via UPI"
                    body = (
                        f"Dear Customer, your {bank} account ending XX{random.randint(1000, 9999)} was debited for INR {amt}.00 "
                        f"towards UPI transaction ref {random.randint(10000000, 99999999)}. If you did not make this transaction, call 24x7 support."
                    )
                elif std_type == "personal":
                    subject = f"Hey {name}, are you free to meet this weekend?"
                    body = (
                        f"Hey! Just checking in to see if you have any plans for dinner or coffee this Saturday evening. "
                        f"Let me know what time works best for you. Talk soon!"
                    )
                elif std_type == "travel":
                    subject = f"Your {brand} ride receipt with driver {name}"
                    body = (
                        f"Thanks for riding with {brand}! Total fare: INR {random.randint(150, 480)}. Payment method: UPI. "
                        f"Distance: 8.4 km. Rate your driver and download the tax invoice from your ride history."
                    )
                else: # subscription
                    subject = f"Your monthly {brand} subscription has been renewed"
                    body = (
                        f"Your subscription plan for {brand} was successfully renewed for INR {random.randint(199, 999)}. "
                        f"Next billing date: 10-November-2026. Manage your subscription anytime in account settings."
                    )

            dataset.append({"subject": subject, "body": body, "label": 0})

    return dataset


# ─────────────────────────────────────────────────────────────────────────────
# 5. Master Dataset Assembler & Cleaner
# ─────────────────────────────────────────────────────────────────────────────

def create_comprehensive_dataset() -> pd.DataFrame:
    """
    Builds the master dataset combining:
    1. Realistic career actionable seed opportunities (label 1)
    2. Exact Glasp newsletter and comprehensive hard-negative seeds (label 0)
    3. Everyday standard non-career seeds (label 0)
    4. Diverse synthetic variations (label 1 and label 0)
    """
    career_seeds = get_career_seed_samples()
    hard_negative_seeds = get_hard_negative_seed_samples()
    standard_non_career_seeds = get_standard_non_career_seed_samples()

    records = []

    # 1. Career seeds
    for s in career_seeds:
        records.append({"subject": s["subject"], "body": s["body"], "label": 1})

    # 2. Hard negative seeds (Glasp newsletter + career-adjacent non-jobs)
    for s in hard_negative_seeds:
        records.append({"subject": s["subject"], "body": s["body"], "label": 0})

    # 3. Standard everyday seeds
    for s in standard_non_career_seeds:
        records.append({"subject": s["subject"], "body": s["body"], "label": 0})

    # 4. Generate balanced variations (approx 220 career, 220 non-career with 50% hard negatives)
    career_vars = generate_synthetic_variations(target_label=1, count=220)
    non_career_vars = generate_synthetic_variations(target_label=0, count=220)

    all_records = records + career_vars + non_career_vars
    df = pd.DataFrame(all_records)
    logger.info(f"Raw dataset assembled: {len(df)} records ({df['label'].value_counts().to_dict()})")
    return df


def clean_and_validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates schema, cleans text, strips nulls/empty entries,
    and performs exact and near-deduplication to prevent cross-split leakage.
    """
    required_cols = {"subject", "body", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required_cols}. Found: {df.columns.tolist()}")

    initial_count = len(df)

    # Drop nulls
    df = df.dropna(subset=["subject", "body", "label"]).copy()

    # Validate labels
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    # Clean whitespace
    df["subject"] = df["subject"].astype(str).str.strip()
    df["body"] = df["body"].astype(str).str.strip()

    # Drop empty rows
    df = df[(df["subject"].str.len() > 0) | (df["body"].str.len() > 0)].copy()

    # Exact deduplication
    df = df.drop_duplicates(subset=["subject", "body"]).copy()

    # Near-duplicate deduplication using canonical signature
    df["dedup_key"] = df.apply(lambda row: text_dedup_key(row["subject"], row["body"]), axis=1)
    df = df.drop_duplicates(subset=["dedup_key"]).copy()
    df = df.drop(columns=["dedup_key"])

    df = df.reset_index(drop=True)
    logger.info(f"Cleaned dataset: {len(df)} records (removed {initial_count - len(df)} duplicates/invalid rows).")
    return df


def prepare_and_split_dataset(
    output_dir: str = "dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Builds, validates, and stratifies the dataset into Train, Val, and Test splits.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = create_comprehensive_dataset()
    df = clean_and_validate_dataset(df)

    # Save cleaned master dataset
    master_csv_path = os.path.join(output_dir, "emails.csv")
    df.to_csv(master_csv_path, index=False)
    logger.info(f"Saved master dataset to {master_csv_path}")

    # Stratified split: Train vs Temp (Val + Test)
    temp_ratio = val_ratio + test_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_ratio,
        stratify=df["label"],
        random_state=random_state
    )

    # Split Temp into Val and Test
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df["label"],
        random_state=random_state
    )

    # Save splits
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    logger.info(f"Dataset Split Summary:")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%) | Positive class: {train_df['label'].sum()} ({train_df['label'].mean()*100:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%) | Positive class: {val_df['label'].sum()} ({val_df['label'].mean()*100:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%) | Positive class: {test_df['label'].sum()} ({test_df['label'].mean()*100:.1f}%)")

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_and_split_dataset()
