"""
Document RAG Compliance Assistant
Interactive demo for document-based question answering using RAG (Retrieval-Augmented Generation).
"""

import gradio as gr
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

# Synthetic compliance document knowledge base
COMPLIANCE_DOCS = {
    "IFRS 17 Overview": """
    IFRS 17 Insurance Contracts is the international accounting standard for insurance contracts.
    
    Key Principles:
    - Measurement: Insurance liabilities measured at current fulfillment value
    - Components: Fulfillment cash flows + Contractual service margin
    - Fulfillment Cash Flows: Estimates of future cash flows, risk adjustment, and discounting
    - Recognition: Revenue recognized as services are provided
    - Effective Date: January 1, 2023
    
    Building Blocks:
    1. Estimates of future cash flows
    2. Adjustment for time value of money (discounting)
    3. Risk adjustment for non-financial risk
    4. Contractual service margin (unearned profit)
    """,
    
    "Fraud Detection Best Practices": """
    Insurance Fraud Detection Best Practices
    
    Detection Methods:
    - Rule-based systems: Predefined criteria and thresholds
    - Anomaly detection: Statistical analysis of unusual patterns
    - Network analysis: Identifying fraud rings and connections
    - Predictive modeling: Machine learning for risk scoring
    
    Red Flags:
    - Claims filed shortly after policy inception
    - Multiple claims in short time period
    - Inconsistent claim details
    - Lack of supporting documentation
    - Claims from high-risk locations
    - Weekend or holiday filing patterns
    
    Investigation Process:
    1. Initial triage and risk scoring
    2. Document review and verification
    3. Interview claimants and witnesses
    4. External data validation
    5. Special investigation unit (SIU) referral if needed
    
    Compliance: All fraud detection must comply with privacy laws and fair claims practices.
    """,
    
    "Claims Reserving Standards": """
    Actuarial Standards for Claims Reserving
    
    Methods:
    - Chain Ladder: Development factor method for projecting ultimate losses
    - Bornhuetter-Ferguson: Combines expected losses with actual development
    - Loss Ratio Method: Based on expected loss ratios
    - Frequency-Severity: Separate analysis of claim counts and amounts
    
    Reserve Components:
    - Case Reserves: Estimated cost of known claims
    - IBNR: Incurred but not reported claims
    - IBNER: Incurred but not enough reserved
    - Reopened Claims: Previously closed claims that reopen
    
    Key Considerations:
    - Development patterns vary by line of business
    - Tail factors for long-tail lines (liability, workers comp)
    - Trend adjustments for inflation and claim cost changes
    - Large loss treatment and catastrophe reserves
    - Discount for time value of money (IFRS 17)
    
    Documentation: All reserve estimates must be documented with clear assumptions and methodology.
    """,
    
    "Data Privacy Regulations": """
    Insurance Data Privacy and Protection
    
    Key Regulations:
    - GDPR (Europe): General Data Protection Regulation
    - CCPA (California): California Consumer Privacy Act
    - HIPAA (US Health): Health Insurance Portability and Accountability Act
    - State insurance privacy laws
    
    Protected Information:
    - Personal identifiable information (PII)
    - Health information (PHI)
    - Financial data
    - Biometric data
    - Location data
    
    Requirements:
    - Consent for data collection and use
    - Right to access personal data
    - Right to deletion (right to be forgotten)
    - Data breach notification
    - Data minimization and purpose limitation
    - Security safeguards and encryption
    
    Insurance-Specific:
    - Fair Credit Reporting Act (FCRA) compliance
    - Unfair discrimination prohibitions
    - Transparent underwriting and rating practices
    - Secure claims data handling
    """,
    
    "Underwriting Guidelines": """
    Insurance Underwriting Compliance Guidelines
    
    Fair Underwriting Practices:
    - Non-discrimination: Cannot discriminate based on protected classes
    - Actuarial justification: Rating factors must be actuarially sound
    - Transparency: Clear disclosure of rating criteria
    - Consistency: Apply guidelines uniformly
    
    Prohibited Factors:
    - Race, color, national origin
    - Religion
    - Gender (in most jurisdictions)
    - Marital status (in some jurisdictions)
    - Genetic information
    
    Required Considerations:
    - Risk assessment based on legitimate factors
    - Loss history and claims experience
    - Coverage limits and deductibles
    - Geographic risk factors (if actuarially justified)
    - Credit-based insurance scores (where permitted)
    
    Documentation:
    - Underwriting decisions must be documented
    - Declination reasons must be provided
    - File documentation for regulatory review
    - Adverse action notices when required
    """,
    
    "Solvency II Requirements": """
    Solvency II Framework (European Insurance Regulation)
    
    Three Pillars:
    1. Quantitative Requirements: Capital requirements and technical provisions
    2. Governance and Risk Management: Internal controls and risk assessment
    3. Disclosure and Transparency: Reporting to supervisors and public
    
    Capital Requirements:
    - SCR (Solvency Capital Requirement): 99.5% VaR over one year
    - MCR (Minimum Capital Requirement): Absolute floor
    - Own Funds: Available capital to meet requirements
    
    Technical Provisions:
    - Best estimate liabilities
    - Risk margin
    - Discounting using risk-free rate
    
    Risk Categories:
    - Underwriting risk (life, non-life, health)
    - Market risk
    - Credit risk
    - Operational risk
    
    Governance:
    - Own Risk and Solvency Assessment (ORSA)
    - Risk management function
    - Actuarial function
    - Internal audit and compliance
    """
}

# Simple keyword-based retrieval (simulating embeddings)
def retrieve_relevant_docs(query: str, top_k: int = 2) -> List[Tuple[str, str, float]]:
    """
    Retrieve relevant documents based on keyword matching.
    
    Args:
        query: User question
        top_k: Number of documents to retrieve
        
    Returns:
        List of (doc_name, doc_content, relevance_score) tuples
    """
    query_lower = query.lower()
    scores = []
    
    for doc_name, doc_content in COMPLIANCE_DOCS.items():
        # Simple keyword matching score
        doc_lower = doc_content.lower()
        
        # Count keyword matches
        keywords = query_lower.split()
        score = sum(1 for keyword in keywords if len(keyword) > 3 and keyword in doc_lower)
        
        # Boost score if doc name matches
        if any(word in doc_name.lower() for word in keywords):
            score += 5
        
        scores.append((doc_name, doc_content, score))
    
    # Sort by score and return top_k
    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]


def generate_answer(query: str, retrieved_docs: List[Tuple[str, str, float]]) -> str:
    """
    Generate answer based on retrieved documents.
    
    Args:
        query: User question
        retrieved_docs: Retrieved documents with scores
        
    Returns:
        Generated answer
    """
    if not retrieved_docs or retrieved_docs[0][2] == 0:
        return """
I couldn't find relevant information in the compliance knowledge base to answer your question.

Please try rephrasing your question or ask about:
- IFRS 17 accounting standards
- Fraud detection practices
- Claims reserving methods
- Data privacy regulations
- Underwriting guidelines
- Solvency II requirements
"""
    
    # Build answer from retrieved documents
    answer = f"**Answer based on compliance documents:**\n\n"
    
    for doc_name, doc_content, score in retrieved_docs:
        if score > 0:
            # Extract relevant sections (simplified - just use first few lines)
            lines = doc_content.strip().split('\n')
            relevant_section = '\n'.join(lines[:15])  # First 15 lines
            
            answer += f"**From: {doc_name}**\n\n{relevant_section}\n\n---\n\n"
    
    answer += """
⚠️ **Disclaimer**: This is a demonstration using synthetic compliance documents. 
For actual compliance guidance, consult official regulations and qualified professionals.
"""
    
    return answer


def rag_query(question: str, num_docs: int = 2) -> Tuple[str, pd.DataFrame]:
    """
    Process RAG query and return answer with source documents.
    
    Args:
        question: User question
        num_docs: Number of documents to retrieve
        
    Returns:
        Tuple of (answer, sources_dataframe)
    """
    # Retrieve relevant documents
    retrieved = retrieve_relevant_docs(question, top_k=num_docs)
    
    # Generate answer
    answer = generate_answer(question, retrieved)
    
    # Create sources dataframe
    sources_data = {
        "Document": [doc[0] for doc in retrieved],
        "Relevance Score": [doc[2] for doc in retrieved],
        "Preview": [doc[1][:200] + "..." for doc in retrieved]
    }
    sources_df = pd.DataFrame(sources_data)
    
    return answer, sources_df


# Create Gradio interface
with gr.Blocks(title="Document RAG Compliance Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📚 Document RAG Compliance Assistant
    
    Ask questions about insurance compliance topics and get answers from our knowledge base.
    
    **Available Topics:**
    - IFRS 17 Accounting Standards
    - Fraud Detection Best Practices
    - Claims Reserving Standards
    - Data Privacy Regulations
    - Underwriting Guidelines
    - Solvency II Requirements
    
    **How it works:**
    1. Enter your question
    2. System retrieves relevant documents (RAG - Retrieval-Augmented Generation)
    3. Answer is generated based on retrieved content
    4. Source documents are shown for transparency
    
    ⚠️ **Demo only** - uses synthetic documents for illustration purposes.
    """)
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the key components of IFRS 17?",
                lines=3
            )
            num_docs_slider = gr.Slider(
                label="Number of Documents to Retrieve",
                minimum=1,
                maximum=5,
                value=2,
                step=1
            )
            submit_btn = gr.Button("Ask Question", variant="primary")
            
            gr.Markdown("### Example Questions:")
            gr.Markdown("""
            - What are the key principles of IFRS 17?
            - How should we detect insurance fraud?
            - What methods are used for claims reserving?
            - What are the data privacy requirements for insurance?
            - What factors are prohibited in underwriting?
            - What are the Solvency II capital requirements?
            """)
    
    gr.Markdown("---")
    
    with gr.Row():
        answer_output = gr.Markdown(label="Answer")
    
    with gr.Row():
        sources_output = gr.Dataframe(
            label="Source Documents",
            headers=["Document", "Relevance Score", "Preview"],
            interactive=False
        )
    
    # Connect button to function
    submit_btn.click(
        fn=rag_query,
        inputs=[question_input, num_docs_slider],
        outputs=[answer_output, sources_output]
    )
    
    # Example questions as buttons
    gr.Markdown("### Quick Examples:")
    
    with gr.Row():
        example1 = gr.Button("IFRS 17 Components")
        example2 = gr.Button("Fraud Red Flags")
        example3 = gr.Button("Reserve Methods")
    
    example1.click(
        fn=lambda: rag_query("What are the key components of IFRS 17 measurement?", 2),
        inputs=[],
        outputs=[answer_output, sources_output]
    )
    
    example2.click(
        fn=lambda: rag_query("What are common fraud detection red flags?", 2),
        inputs=[],
        outputs=[answer_output, sources_output]
    )
    
    example3.click(
        fn=lambda: rag_query("What methods are used for claims reserving?", 2),
        inputs=[],
        outputs=[answer_output, sources_output]
    )
    
    gr.Markdown("""
    ---
    
    ### About RAG (Retrieval-Augmented Generation)
    
    RAG is a technique that combines:
    1. **Retrieval**: Finding relevant documents from a knowledge base
    2. **Generation**: Creating answers based on retrieved content
    
    **Benefits:**
    - Grounded in source documents (reduces hallucination)
    - Transparent (shows sources)
    - Updatable (add new documents without retraining)
    - Domain-specific (uses your organization's knowledge)
    
    **This Demo:**
    - Uses simple keyword matching (real systems use embeddings)
    - Synthetic compliance documents (not real regulations)
    - Template-based generation (real systems use LLMs)
    
    ---
    
    **Built by Qoder for Vercept** | All data synthetic | Advisory only
    """)

if __name__ == "__main__":
    demo.launch()
