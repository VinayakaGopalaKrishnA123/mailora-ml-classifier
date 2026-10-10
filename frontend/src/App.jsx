import React, { useState, useEffect } from 'react';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Realistic sample emails for immediate user testing
const SAMPLES = [
  {
    name: 'Internship Offer',
    subject: 'Summer 2027 Software Engineering Internship - Google',
    body: 'We are thrilled to invite you to apply for our Summer 2027 SWE Internship cohort. Work alongside core engineering teams on distributed systems and cloud infrastructure. Open to pre-final year students. Deadline: Nov 15.',
  },
  {
    name: 'Interview Invite',
    subject: 'Interview Schedule: Technical Round 1 - Backend Developer at Cred',
    body: 'Hi Candidate, Congratulations on clearing our online assessment! We have scheduled your 60-minute technical interview for Thursday at 3:00 PM IST via Google Meet: https://meet.google.com/xyz-cred-eng.',
  },
  {
    name: 'Application Status (Rejection)',
    subject: 'Update regarding your job application for Software Engineer at Stripe',
    body: 'Dear Applicant, Thank you for interviewing with us for the Software Engineer role. After careful review, we will not be moving forward with your candidacy for this position. We wish you success in your job search.',
  },
  {
    name: 'Shopping Delivery (Non-Career)',
    subject: 'Your Amazon.in order #402-9182741 has been dispatched!',
    body: 'Your package containing Logitech Wireless Mouse is out for delivery with courier BlueDart. Expected delivery by 8 PM today. Track live on Amazon app.',
  },
  {
    name: 'Credit Card Bill (Non-Career)',
    subject: 'HDFC Bank: Monthly e-Statement for Credit Card ending 4092',
    body: 'Dear Customer, Your monthly credit card statement for October is ready. Total Amount Due: INR 14,820.50. Minimum Due: INR 1,200. Due Date: 28-Oct-2026.',
  },
];

export default function App() {
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState({ online: false, version: '', checking: true });

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        setApiHealth({ online: true, version: data.model_version || '1.0.0', checking: false });
      } else {
        setApiHealth({ online: false, version: '', checking: false });
      }
    } catch {
      setApiHealth({ online: false, version: '', checking: false });
    }
  };

  const handleSelectSample = (sample) => {
    setSubject(sample.subject);
    setBody(sample.body);
    setPrediction(null);
    setError(null);
  };

  const handleClear = () => {
    setSubject('');
    setBody('');
    setPrediction(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validation
    const cleanSubject = subject.trim();
    const cleanBody = body.trim();

    if (!cleanSubject && !cleanBody) {
      setError('Please provide an email subject or email body to classify.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          subject: cleanSubject,
          body: cleanBody,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server responded with HTTP status ${response.status}`);
      }

      const result = await response.json();
      setPrediction(result);
    } catch (err) {
      console.error('Classification request failed:', err);
      setError(
        err.message.includes('Failed to fetch') || err.message.includes('NetworkError')
          ? `Could not reach the classification server at ${API_BASE_URL}. Ensure the backend service is running (python -m uvicorn src.api:app --host 127.0.0.1 --port 8000).`
          : `Classification error: ${err.message}`
      );
    } finally {
      setIsLoading(false);
    }
  };

  const isCareer = prediction && prediction.is_career_related;
  const confidencePercent = prediction ? (prediction.confidence * 100).toFixed(1) : null;
  const careerProbPercent = prediction && prediction.career_probability !== undefined 
    ? (prediction.career_probability * 100).toFixed(1) 
    : null;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="brand-wrapper">
          <svg className="brand-logo" viewBox="0 0 24 24" fill="currentColor">
            <path d="M1.5 8.67v8.58a3 3 0 003 3h15a3 3 0 003-3V8.67l-8.928 5.493a3 3 0 01-3.144 0L1.5 8.67z" />
            <path d="M22.5 6.908V6.75a3 3 0 00-3-3h-15a3 3 0 00-3 3v.158l9.714 5.978a1.5 1.5 0 001.572 0L22.5 6.908z" />
          </svg>
          <span className="brand-name">Mailora</span>
          <span className="badge-tag">NLP Engine</span>
        </div>

        <h1 className="main-title">AI Email Classifier</h1>
        <p className="subtitle">Discover career opportunities hidden in your emails.</p>

        {/* API Health Pill */}
        <div className="status-bar" title="API Status">
          <span className={`status-dot ${apiHealth.online ? 'online' : 'offline'}`}></span>
          <span>
            {apiHealth.checking
              ? 'Connecting to ML API...'
              : apiHealth.online
              ? `ML API Connected (v${apiHealth.version})`
              : 'ML API Offline (Check port 8000)'}
          </span>
        </div>
      </header>

      {/* Main Form Card */}
      <main className="main-card">
        {/* Sample Loaders */}
        <div className="samples-wrapper">
          <span className="samples-label">Quick Test Examples:</span>
          <div className="samples-chips">
            {SAMPLES.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                className="chip-btn"
                onClick={() => handleSelectSample(sample)}
                disabled={isLoading}
              >
                {sample.name}
              </button>
            ))}
          </div>
        </div>

        <form onSubmit={handleSubmit}>
          {/* Subject Field */}
          <div className="form-group">
            <label htmlFor="email-subject" className="form-label">
              <span>Email Subject</span>
              <span className="char-count">{subject.length} chars</span>
            </label>
            <input
              id="email-subject"
              type="text"
              className="input-field"
              placeholder="e.g. Software Engineer Internship - Applications Open"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              disabled={isLoading}
            />
          </div>

          {/* Email Body Field */}
          <div className="form-group">
            <label htmlFor="email-body" className="form-label">
              <span>Email Body</span>
              <span className="char-count">{body.length} chars</span>
            </label>
            <textarea
              id="email-body"
              className="textarea-field"
              placeholder="Paste email body, job description, or message text here..."
              value={body}
              onChange={(e) => setBody(e.target.value)}
              disabled={isLoading}
            />
          </div>

          {/* Action Buttons */}
          <div className="actions-row">
            <button
              id="classify-button"
              type="submit"
              className="btn-primary"
              disabled={isLoading || (!subject.trim() && !body.trim())}
            >
              {isLoading ? (
                <>
                  <span className="spinner"></span>
                  <span>Classifying Email...</span>
                </>
              ) : (
                <>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 2L11 13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                  <span>Classify Email</span>
                </>
              )}
            </button>

            <button
              type="button"
              className="btn-secondary"
              onClick={handleClear}
              disabled={isLoading || (!subject && !body && !prediction && !error)}
            >
              Clear
            </button>
          </div>
        </form>

        {/* Error Banner */}
        {error && (
          <div className="error-banner" role="alert">
            <span className="error-icon">⚠️</span>
            <div>{error}</div>
          </div>
        )}

        {/* Real Prediction Results Section */}
        {prediction && (
          <section className={`result-card ${isCareer ? 'career' : 'non-career'}`} aria-live="polite">
            <div className="result-header">
              <div className="indicator-badge">
                <span className="status-icon">
                  {isCareer ? '✓' : '•'}
                </span>
                <span>
                  {isCareer ? 'Career-Related Email' : 'Non-Career-Related Email'}
                </span>
              </div>

              {confidencePercent && (
                <div className="confidence-pill" title="Model Confidence Score">
                  <span className="confidence-label">Confidence:</span>
                  <span className="confidence-val">{confidencePercent}%</span>
                </div>
              )}
            </div>

            <p className="result-description">
              {isCareer
                ? 'This email contains job postings, internship invitations, technical assessments, interviews, or application status updates tracked by Mailora.'
                : 'This email appears to be general correspondence, a personal message, e-commerce order, banking statement, social media alert, or newsletter.'}
            </p>

            {/* Model Metadata */}
            <div className="meta-grid">
              <div className="meta-item">
                <div className="meta-title">Classification</div>
                <div className="meta-content">{prediction.label}</div>
              </div>

              {careerProbPercent && (
                <div className="meta-item">
                  <div className="meta-title">Career Probability</div>
                  <div className="meta-content">{careerProbPercent}%</div>
                </div>
              )}

              <div className="meta-item">
                <div className="meta-title">Inference Latency</div>
                <div className="meta-content">{prediction.latency_ms} ms</div>
              </div>

              <div className="meta-item">
                <div className="meta-title">Model Version</div>
                <div className="meta-content">{prediction.model_version}</div>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
