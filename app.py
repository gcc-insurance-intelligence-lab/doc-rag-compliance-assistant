"""
Document RAG Compliance Assistant
Retrieval-Augmented Generation system for insurance policy compliance questions.
Loads synthetic policy clauses and answers questions using semantic search.
"""

import gradio as gr
import numpy as np
from typing import List, Tuple
import os

# Try to import sentence transformers, fall back to TF-IDF if not available
try:
    from sentence_transformers import SentenceTransformer
    USE_TRANSFORMERS = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    USE_TRANSFORMERS = False


class PolicyRAGEngine:
    """RAG engine for policy document retrieval and question answering."""
    
    def __init__(self, policy_file_path: str):
        self.policy_file_path = policy_file_path
        self.clauses = []
        self.clause_titles = []
        self.embeddings = None
        self.model = None
        self.vectorizer = None
        
        # Load policy clauses
        self._load_policy_clauses()
        
        # Initialize embedding model
        self._initialize_embeddings()
    
    def _load_policy_clauses(self):
        """Load policy clauses from text file."""
        if not os.path.exists(self.policy_file_path):
            # Fallback to relative path
            self.policy_file_path = "../insurance-datasets-synthetic/data/policy_clauses_snippets.txt"
        
        with open(self.policy_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by clause headers
        sections = content.split('CLAUSE ')
        
        for section in sections[1:]:  # Skip first empty section
            lines = section.strip().split('\n', 1)
            if len(lines) == 2:
                title = "CLAUSE " + lines[0].strip()
                text = lines[1].strip()
                self.clause_titles.append(title)
                self.clauses.append(text)
    
    def _initialize_embeddings(self):
        """Initialize embedding model (sentence transformers or TF-IDF)."""
        global USE_TRANSFORMERS
        if USE_TRANSFORMERS:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.embeddings = self.model.encode(self.clauses)
            except Exception as e:
                print(f"Error loading sentence transformers: {e}. Falling back to TF-IDF.")
                USE_TRANSFORMERS = False
        
        if not USE_TRANSFORMERS:
            # Fallback to TF-IDF
            self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            self.embeddings = self.vectorizer.fit_transform(self.clauses)
    
    def retrieve_relevant_clause(self, question: str, top_k: int = 1) -> List[Tuple[str, str, float]]:
        """Retrieve most relevant policy clause for a question."""
        if USE_TRANSFORMERS and self.model is not None:
            # Use sentence transformers
            question_embedding = self.model.encode([question])[0]
            similarities = np.dot(self.embeddings, question_embedding)
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((
                    self.clause_titles[idx],
                    self.clauses[idx],
                    float(similarities[idx])
                ))
            return results
        else:
            # Use TF-IDF
            question_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(question_vec, self.embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append((
                    self.clause_titles[idx],
                    self.clauses[idx],
                    float(similarities[idx])
                ))
            return results
    
    def answer_question(self, question: str) -> Tuple[str, str, float, str, str]:
        """Answer a question using RAG approach."""
        # Check for out-of-scope questions
        out_of_scope_keywords = [
            'pricing', 'premium calculation', 'underwriting decision',
            'approve', 'deny', 'legal advice', 'tax', 'investment',
            'medical diagnosis', 'claim amount', 'reserve amount'
        ]
        
        question_lower = question.lower()
        for keyword in out_of_scope_keywords:
            if keyword in question_lower:
                return (
                    "Out of Scope",
                    "This question is beyond the scope of this compliance assistant. This tool only provides information about policy clauses and general compliance concepts. For pricing, underwriting, legal advice, or specific claim decisions, please consult appropriate professionals.",
                    0.0,
                    "This question is out of scope for this tool.",
                    "⚠️ This question requires human expert consultation."
                )
        
        # Retrieve most relevant clause
        results = self.retrieve_relevant_clause(question, top_k=1)
        
        if not results:
            return (
                "No Match",
                "I couldn't find a relevant policy clause for your question.",
                0.0,
                "No matching clause found.",
                "⚠️ Please consult policy documentation or compliance team."
            )
        
        title, clause_text, similarity = results[0]
        
        # Generate explanation
        explanation = self._generate_explanation(question, title, clause_text, similarity)
        
        # Generate warning
        warning = self._generate_warning(similarity)
        
        return title, clause_text, similarity, explanation, warning
    
    def _generate_explanation(self, question: str, title: str, clause: str, similarity: float) -> str:
        """Generate explanation for the retrieved clause."""
        if similarity < 0.3:
            return f"The most relevant clause found is '{title}', but the similarity score is low ({similarity:.2f}). This clause may not directly answer your question. Please review the clause text carefully and consult compliance experts if needed."
        elif similarity < 0.6:
            return f"Found '{title}' with moderate relevance ({similarity:.2f}). This clause may contain information related to your question. Please review the full clause text and verify with compliance team."
        else:
            return f"Found '{title}' with high relevance ({similarity:.2f}). This clause appears to be directly related to your question. However, always verify with compliance experts before making decisions."
    
    def _generate_warning(self, similarity: float) -> str:
        """Generate appropriate warning based on similarity score."""
        base_warning = "⚠️ **IMPORTANT**: This is an automated retrieval system. "
        
        if similarity < 0.3:
            return base_warning + "The match quality is LOW. This response may not be accurate. **Human review is REQUIRED** before using this information."
        elif similarity < 0.6:
            return base_warning + "The match quality is MODERATE. **Human review is REQUIRED** to verify accuracy and applicability."
        else:
            return base_warning + "Even with high match quality, **human review is REQUIRED**. Never make compliance decisions based solely on automated systems."


# Initialize RAG engine
POLICY_FILE = "../insurance-datasets-synthetic/data/policy_clauses_snippets.txt"
rag_engine = PolicyRAGEngine(POLICY_FILE)


def process_question(question: str) -> Tuple[str, str, str, str, str]:
    """Process user question and return formatted response."""
    if not question or len(question.strip()) < 5:
        return (
            "Invalid Input",
            "Please enter a valid question (at least 5 characters).",
            "N/A",
            "",
            "⚠️ Please provide a complete question."
        )
    
    title, clause_text, similarity, explanation, warning = rag_engine.answer_question(question)
    
    # Format similarity score
    similarity_str = f"{similarity:.3f}" if similarity > 0 else "N/A"
    
    return title, clause_text, similarity_str, explanation, warning


# Create Gradio interface
with gr.Blocks(title="Document RAG Compliance Assistant") as demo:
    gr.Markdown("""
    # 📚 Document RAG Compliance Assistant
    
    ## ⚠️ CRITICAL DISCLAIMER
    
    **This tool is for EDUCATIONAL and DEMONSTRATION purposes ONLY.**
    
    - **NOT for production compliance decisions**
    - **NOT a substitute for legal or compliance professionals**
    - **All outputs require human expert validation**
    - **No liability for decisions based on this tool**
    
    This system uses Retrieval-Augmented Generation (RAG) to find relevant policy clauses
    based on your questions. It searches through synthetic policy documents and returns
    the most relevant clause with an explanation.
    
    ### How It Works:
    1. Enter your compliance or policy question
    2. The system searches through policy clauses using semantic similarity
    3. Returns the most relevant clause with similarity score
    4. Provides explanation and mandatory human review warning
    
    ### Out-of-Scope Questions:
    This tool does NOT answer questions about:
    - Pricing or premium calculations
    - Underwriting decisions
    - Legal advice
    - Specific claim amounts or approvals
    - Tax or investment advice
    """)
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the deductible requirements? What happens if I cancel my policy?",
                lines=3
            )
            submit_btn = gr.Button("Search Policy Clauses", variant="primary")
            clear_btn = gr.Button("Clear")
    
    with gr.Row():
        with gr.Column():
            clause_title = gr.Textbox(label="Relevant Clause", interactive=False)
            similarity_score = gr.Textbox(label="Similarity Score", interactive=False)
    
    with gr.Row():
        clause_text = gr.Textbox(
            label="Clause Text",
            lines=8,
            interactive=False
        )
    
    with gr.Row():
        explanation = gr.Textbox(
            label="Explanation",
            lines=4,
            interactive=False
        )
    
    with gr.Row():
        warning = gr.Markdown("### ⚠️ Human Review Required")
    
    # Example questions
    gr.Examples(
        examples=[
            ["What are the coverage limitations?"],
            ["How do I file a claim?"],
            ["What is the deductible policy?"],
            ["Can the policy be cancelled?"],
            ["What happens in case of fraud?"],
            ["What are my duties after a loss?"]
        ],
        inputs=question_input
    )
    
    # Event handlers
    submit_btn.click(
        fn=process_question,
        inputs=[question_input],
        outputs=[clause_title, clause_text, similarity_score, explanation, warning]
    )
    
    clear_btn.click(
        fn=lambda: ("", "", "", "", ""),
        inputs=[],
        outputs=[clause_title, clause_text, similarity_score, explanation, warning]
    )
    
    gr.Markdown("""
    ---
    
    ### 🛡️ Compliance & Safety Notes:
    
    - **Synthetic Data Only**: All policy clauses are fabricated for demonstration
    - **No Real Policies**: Does not contain actual insurance product information
    - **Human-in-the-Loop**: All outputs must be reviewed by compliance professionals
    - **Educational Purpose**: For learning RAG concepts, not production use
    - **No Guarantees**: Similarity scores do not guarantee accuracy or applicability
    
    **Always consult qualified compliance, legal, and insurance professionals for actual policy interpretation.**
    """)

if __name__ == "__main__":
    demo.launch()
