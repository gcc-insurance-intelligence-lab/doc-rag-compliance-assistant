# Model Card: Document RAG Compliance Assistant

## Model Details

### Model Description

This is a **Retrieval-Augmented Generation (RAG) demonstration system** for insurance compliance question answering. It uses keyword-based document retrieval and template-based answer generation (not a trained ML model).

- **Developed by:** Qoder for Vercept
- **Model type:** RAG system (retrieval + generation pipeline)
- **Retrieval method:** Keyword matching (demo - production would use embeddings)
- **Generation method:** Template-based (demo - production would use LLM)
- **Language:** Python
- **License:** MIT

### Model Sources

- **Repository:** Hugging Face Spaces
- **Demo:** Interactive Gradio interface

## Uses

### Direct Use

This tool is designed for:

- **Educational purposes**: Learning about RAG systems
- **Demonstration**: Showcasing document-based Q&A
- **Prototyping**: Testing RAG workflows
- **Training**: Teaching compliance concepts
- **Concept validation**: Understanding retrieval-generation pipelines

### Downstream Use

Not applicable - this is a standalone demonstration tool.

### Out-of-Scope Use

⚠️ **This tool should NOT be used for:**

- Actual compliance guidance or legal advice
- Regulatory filings or submissions
- Audit support or documentation
- Legal decision-making
- Official policy interpretation
- Any situation requiring professional legal/compliance counsel
- Production knowledge management systems
- Customer-facing compliance support

## Bias, Risks, and Limitations

### Known Limitations

**Retrieval Limitations:**
1. **Keyword Matching Only**: Doesn't understand semantic similarity
2. **No Embeddings**: Can't capture meaning beyond exact word matches
3. **Simple Scoring**: Basic term frequency, no TF-IDF or BM25
4. **No Reranking**: Doesn't refine initial retrieval results
5. **Small Knowledge Base**: Only 6 synthetic documents
6. **No Chunking**: Uses entire documents, not optimized sections

**Generation Limitations:**
1. **Template-Based**: Not true natural language generation
2. **No LLM**: Doesn't synthesize or reason over information
3. **No Citations**: Can't cite specific sentences or paragraphs
4. **No Summarization**: Just displays document sections
5. **No Multi-hop Reasoning**: Can't combine information across documents
6. **No Conversation**: Doesn't maintain context or follow-ups

**Content Limitations:**
1. **Synthetic Documents**: Not real regulations or official guidance
2. **Simplified Content**: Actual compliance is far more complex
3. **Static Knowledge**: Can't update or learn from interactions
4. **Limited Coverage**: Only 6 topics, not comprehensive
5. **No Versioning**: Doesn't track document versions or updates

### Potential Risks

**Misinformation Risk:**
- Users might mistake synthetic content for real guidance
- Simplified explanations may omit critical nuances
- Keyword matching may retrieve irrelevant documents

**Over-reliance Risk:**
- Users might use for actual compliance decisions
- May substitute for professional legal counsel
- Could lead to compliance violations if misused

**Technical Risks:**
- Poor retrieval quality with ambiguous queries
- Missing relevant information due to keyword mismatch
- No validation of answer accuracy

### Recommendations

Users should:

- Understand this is a **demonstration only**
- Never use for actual compliance or legal decisions
- Consult qualified professionals for real guidance
- Recognize the difference between this demo and production RAG systems
- Review official regulations and standards directly
- Verify all information with authoritative sources

## How to Get Started

```python
import gradio as gr
from app import demo

# Launch the interface
demo.launch()
```

Or visit the Hugging Face Space to use the interactive demo.

## Training Details

Not applicable - this is a rule-based retrieval and template-based generation system, not a trained model.

## Evaluation

### Testing Data, Factors & Metrics

No formal evaluation has been conducted. The system uses deterministic keyword matching and template generation without statistical validation.

**For production RAG systems, typical metrics include:**

**Retrieval Metrics:**
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)
- Recall@K
- Precision@K

**Generation Metrics:**
- ROUGE (overlap with reference answers)
- BLEU (n-gram precision)
- BERTScore (semantic similarity)
- Human evaluation (accuracy, helpfulness, faithfulness)

**End-to-End Metrics:**
- Answer accuracy
- Source attribution accuracy
- Hallucination rate
- User satisfaction

### Results

Not applicable.

## Environmental Impact

Minimal - this is a lightweight keyword-based system with no model training or inference.

## Technical Specifications

### System Architecture

**Pipeline:**

```
User Query
    ↓
[1] Query Processing
    - Lowercase conversion
    - Keyword extraction
    ↓
[2] Document Retrieval
    - Keyword matching against all documents
    - Relevance scoring (term frequency + name boost)
    - Ranking by score
    ↓
[3] Top-K Selection
    - Select top N documents (default: 2)
    - Filter out zero-score documents
    ↓
[4] Answer Generation
    - Extract relevant sections from documents
    - Format with document attribution
    - Add disclaimer
    ↓
Answer + Source Documents
```

### Retrieval Algorithm

**Scoring Function:**

```python
score = keyword_matches + (5 if doc_name_match else 0)
```

Where:
- `keyword_matches` = count of query keywords (>3 chars) found in document
- `doc_name_match` = bonus if query word appears in document name

**Limitations of this approach:**
- No term weighting (all keywords equal)
- No inverse document frequency (IDF)
- No semantic understanding
- No handling of synonyms or related terms

**Production alternative:**
- Sentence embeddings (e.g., all-MiniLM-L6-v2)
- Vector similarity (cosine, dot product)
- Hybrid search (BM25 + dense retrieval)

### Generation Algorithm

**Current (Template-based):**

```python
answer = "**From: {doc_name}**\n\n{doc_section}"
```

**Production alternative:**

```python
prompt = f"""
Answer the question based on the following documents:

{retrieved_docs}

Question: {query}

Answer:
"""

answer = llm.generate(prompt)
```

### Knowledge Base

**Documents (6 total):**
1. IFRS 17 Overview
2. Fraud Detection Best Practices
3. Claims Reserving Standards
4. Data Privacy Regulations
5. Underwriting Guidelines
6. Solvency II Requirements

**Document characteristics:**
- Synthetic/illustrative content
- 200-400 words each
- Structured with headers and bullet points
- Simplified from real regulations

### Compute Infrastructure

**Requirements**: Minimal - runs on CPU

**Dependencies**:
- Python 3.9+
- Gradio 4.44.0
- Pandas 2.1.4
- NumPy 1.26.2

**No GPU required** - no neural models used

## Model Card Contact

For questions or feedback, contact Vercept.

## Glossary

- **RAG**: Retrieval-Augmented Generation - combining document retrieval with text generation
- **Retrieval**: Finding relevant documents from a knowledge base
- **Generation**: Creating answers based on retrieved content
- **Embeddings**: Vector representations of text for semantic similarity
- **Vector Database**: Database optimized for similarity search (e.g., FAISS, Pinecone)
- **Semantic Search**: Finding documents by meaning, not just keywords
- **Hallucination**: When a model generates unsupported or false information
- **Grounding**: Basing answers on retrieved source documents
- **Citation**: Attributing information to specific sources
- **Context Window**: Maximum text length a model can process

## RAG System Comparison

### This Demo vs. Production RAG

| Component | This Demo | Production RAG |
|-----------|-----------|----------------|
| **Retrieval** | Keyword matching | Semantic embeddings (sentence-transformers) |
| **Vector DB** | None | FAISS, Pinecone, Weaviate, Qdrant |
| **Reranking** | None | Cross-encoder models |
| **Generation** | Templates | LLMs (GPT-4, Claude, Llama) |
| **Citations** | Document-level | Sentence/paragraph-level |
| **Conversation** | None | Multi-turn with history |
| **Evaluation** | None | Automated metrics + human eval |
| **Knowledge Base** | 6 static docs | Thousands of docs, regularly updated |
| **Chunking** | Full documents | Optimized chunks (512-1024 tokens) |
| **Metadata** | None | Dates, authors, versions, tags |

## Production RAG Best Practices

For building a real RAG system:

**1. Document Processing:**
- Chunk documents into optimal sizes (512-1024 tokens)
- Preserve context across chunks
- Add metadata (source, date, version)
- Handle multiple formats (PDF, HTML, DOCX)

**2. Retrieval:**
- Use semantic embeddings (not keywords)
- Implement hybrid search (dense + sparse)
- Add reranking for precision
- Tune top-K parameter

**3. Generation:**
- Use capable LLMs (GPT-4, Claude, Llama-70B+)
- Engineer prompts for faithfulness
- Implement citation tracking
- Add fact verification

**4. Evaluation:**
- Track retrieval metrics (MRR, NDCG)
- Measure generation quality (ROUGE, human eval)
- Monitor hallucination rate
- A/B test improvements

**5. Safety:**
- Validate sources before adding to knowledge base
- Implement content filtering
- Add disclaimers for advisory content
- Log queries for audit

## Model Card Authors

Qoder (Vercept)

## Disclaimer

⚠️ **CRITICAL NOTICE**:

This project models generic insurance concepts common in GCC markets. All datasets are synthetic and made-up for demonstration and research purposes. No proprietary pricing, underwriting rules, policy wording, or confidential logic was used. Outputs are illustrative only and require human review. Not to be used for any pricing, reserving, claim approval, or policy issuance.

## Human-In-The-Loop

No AI component here issues approvals, denials, or financial outcomes. All outputs require human verification and decision-making.

---

This is a **simplified educational demonstration** using synthetic compliance documents. It is **not suitable for**:

- Actual compliance guidance or legal advice
- Regulatory filings or submissions
- Audit support or official documentation
- Any decision-making affecting real compliance obligations

All documents are **synthetic and illustrative only**. They do not reflect actual regulations, official guidance, or legal requirements.

**For actual compliance needs, consult qualified legal and compliance professionals and refer to official regulatory sources.**
