---
title: Document RAG Compliance Assistant
emoji: 📚
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# Document RAG Compliance Assistant

## Overview

An interactive demonstration of Retrieval-Augmented Generation (RAG) for insurance compliance question answering. This tool retrieves relevant documents from a knowledge base and generates answers based on the retrieved content.

---

## Disclaimer

This project models generic insurance concepts common in GCC markets. All datasets are synthetic and made-up for demonstration and research purposes. No proprietary pricing, underwriting rules, policy wording, or confidential logic was used. Outputs are illustrative only and require human review. Not to be used for any pricing, reserving, claim approval, or policy issuance.

## Human-In-The-Loop

No AI component here issues approvals, denials, or financial outcomes. All outputs require human verification and decision-making.

---

## Features

- **Document Retrieval**: Finds relevant compliance documents based on user questions
- **Answer Generation**: Creates answers grounded in source documents
- **Source Transparency**: Shows which documents were used to generate answers
- **Multiple Topics**: Covers IFRS 17, fraud detection, reserving, privacy, underwriting, and Solvency II
- **Interactive Interface**: Ask questions in natural language
- **Relevance Scoring**: Ranks documents by relevance to the query

## Knowledge Base Topics

### 1. IFRS 17 Overview
- Measurement principles
- Fulfillment cash flows
- Contractual service margin
- Recognition and effective date

### 2. Fraud Detection Best Practices
- Detection methods (rules, anomaly detection, ML)
- Common red flags
- Investigation process
- Compliance considerations

### 3. Claims Reserving Standards
- Actuarial methods (chain ladder, BF, etc.)
- Reserve components (case, IBNR, IBNER)
- Key considerations and documentation

### 4. Data Privacy Regulations
- GDPR, CCPA, HIPAA
- Protected information types
- Requirements and rights
- Insurance-specific rules

### 5. Underwriting Guidelines
- Fair underwriting practices
- Prohibited discrimination factors
- Required considerations
- Documentation requirements

### 6. Solvency II Requirements
- Three pillar framework
- Capital requirements (SCR, MCR)
- Technical provisions
- Governance and risk management

## How It Works

### RAG Architecture

```
User Question
    ↓
Document Retrieval (keyword matching)
    ↓
Relevance Scoring
    ↓
Top-K Document Selection
    ↓
Answer Generation (template-based)
    ↓
Answer + Sources
```

### Retrieval Process

1. **Query Processing**: User question is analyzed for keywords
2. **Document Scoring**: Each document is scored based on keyword matches
3. **Ranking**: Documents are ranked by relevance score
4. **Selection**: Top-K most relevant documents are selected
5. **Source Display**: Retrieved documents are shown with scores

### Answer Generation

1. **Context Assembly**: Relevant sections from retrieved documents are combined
2. **Answer Formatting**: Information is structured into a readable answer
3. **Source Attribution**: Each piece of information is linked to its source document
4. **Disclaimer Addition**: Advisory notice is appended

## Usage

### Asking Questions

**Example Questions:**

- "What are the key components of IFRS 17?"
- "How should we detect insurance fraud?"
- "What methods are used for claims reserving?"
- "What are the data privacy requirements?"
- "What factors are prohibited in underwriting?"
- "What are the Solvency II capital requirements?"

### Adjusting Parameters

- **Number of Documents**: Control how many documents to retrieve (1-5)
- More documents = broader context but potentially less focused answers

### Interpreting Results

**Answer Section:**
- Shows information extracted from relevant documents
- Organized by source document
- Includes document name for each section

**Source Documents Table:**
- Lists retrieved documents
- Shows relevance scores
- Provides preview of document content

## Technical Implementation

### Retrieval Method

**Current (Demo):**
- Simple keyword matching
- Term frequency scoring
- Document name boosting

**Production RAG Systems:**
- Semantic embeddings (e.g., sentence-transformers)
- Vector databases (e.g., FAISS, Pinecone, Weaviate)
- Dense retrieval with neural models
- Hybrid search (keyword + semantic)

### Generation Method

**Current (Demo):**
- Template-based extraction
- Direct document section display
- Rule-based formatting

**Production RAG Systems:**
- Large language models (GPT, Claude, Llama)
- Prompt engineering with retrieved context
- Citation and source tracking
- Hallucination detection

## Limitations

### Current Demo Limitations

1. **Simplified Retrieval**: Keyword matching vs. semantic understanding
2. **No Embeddings**: Doesn't capture semantic similarity
3. **Template Generation**: Not true natural language generation
4. **Small Knowledge Base**: Only 6 synthetic documents
5. **No Context Window**: Doesn't maintain conversation history
6. **No Citation Granularity**: Can't cite specific sentences
7. **Static Content**: Can't update knowledge base dynamically

### General RAG Limitations

- **Retrieval Quality**: Answers depend on retrieval accuracy
- **Knowledge Coverage**: Limited to documents in knowledge base
- **Hallucination Risk**: LLMs may still generate unsupported claims
- **Context Length**: Limited by model context window
- **Latency**: Retrieval + generation adds processing time

## Compliance & Safety

⚠️ **IMPORTANT DISCLAIMERS**:

- This is a **demonstration tool only** using synthetic documents
- **Not suitable for actual compliance guidance** or legal advice
- All documents are **illustrative and simplified**
- Does not replace professional legal or compliance counsel
- No real regulations, policies, or official guidance included
- All outputs are **advisory only**
- Not intended for regulatory filings or compliance decisions

## Advantages of RAG

### vs. Fine-tuning

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Knowledge Updates** | Easy - add documents | Hard - retrain model |
| **Transparency** | High - shows sources | Low - black box |
| **Hallucination** | Lower - grounded in docs | Higher - memorized patterns |
| **Cost** | Lower - no retraining | Higher - GPU training |
| **Domain Specificity** | High - your documents | Medium - depends on data |

### Use Cases

**RAG is ideal for:**
- Compliance and regulatory Q&A
- Internal knowledge bases
- Customer support with documentation
- Research and literature review
- Policy and procedure guidance

**Fine-tuning is better for:**
- Specific writing styles
- Domain-specific language
- Task-specific behaviors
- When sources aren't needed

## Future Enhancements

For a production system, consider:

1. **Better Retrieval**:
   - Semantic embeddings (sentence-transformers)
   - Vector database (FAISS, Pinecone)
   - Hybrid search (BM25 + dense)
   - Reranking models

2. **Better Generation**:
   - LLM integration (GPT-4, Claude, Llama)
   - Prompt engineering
   - Citation tracking
   - Fact verification

3. **Better UX**:
   - Conversation history
   - Follow-up questions
   - Document upload
   - Export answers

4. **Better Evaluation**:
   - Retrieval metrics (MRR, NDCG)
   - Generation metrics (ROUGE, BLEU)
   - Human evaluation
   - A/B testing

## Technical Details

- **Framework**: Gradio 4.44.0
- **Language**: Python 3.9+
- **Dependencies**: pandas, numpy
- **Retrieval**: Keyword-based (demo)
- **Generation**: Template-based (demo)

## Example Workflow

1. User asks: "What are IFRS 17 building blocks?"
2. System retrieves "IFRS 17 Overview" document (high relevance)
3. System extracts relevant section about building blocks
4. Answer shows: 4 building blocks with descriptions
5. Source table shows: IFRS 17 Overview with relevance score

## License

MIT License

---

**Built by Qoder for Vercept**

---

**For educational and demonstration purposes only**
