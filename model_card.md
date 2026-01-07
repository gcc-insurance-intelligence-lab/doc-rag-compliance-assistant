# Model Card: Document RAG Compliance Assistant

## Disclaimer

⚠️ **CRITICAL NOTICE:**

This tool is for **educational and demonstration purposes ONLY**. It does NOT provide legal advice, compliance guidance, or regulatory support.

- **NOT for production compliance decisions**
- **NOT a substitute for legal or compliance professionals**
- **All outputs require human expert validation**
- **No liability for decisions based on this tool**

## Model Details

### Model Description

This is a **Retrieval-Augmented Generation (RAG) demonstration system** for insurance policy compliance question answering. It uses semantic search (sentence transformers or TF-IDF fallback) to find relevant policy clauses and provides explanations with mandatory human review warnings.

- **Developed by:** Qoder for Vercept
- **Model type:** RAG system (retrieval + explanation pipeline)
- **Retrieval method:** Sentence transformers (all-MiniLM-L6-v2) with TF-IDF fallback
- **Generation method:** Rule-based explanation generation
- **Language:** Python
- **License:** MIT

### Model Sources

- **Repository:** Hugging Face Spaces
- **Demo:** Interactive Gradio interface
- **Knowledge Base:** Synthetic policy clauses from policy_clauses_snippets.txt

## Uses

### Direct Use

This tool is designed for:

- **Educational purposes**: Learning about RAG systems and semantic search
- **Demonstration**: Showcasing document-based policy Q&A
- **Prototyping**: Testing RAG workflows for compliance
- **Training**: Teaching policy interpretation concepts
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
- Pricing, underwriting, or claim decisions

## Bias, Risks, and Limitations

### Known Limitations

**Retrieval Limitations:**
1. **Limited Knowledge Base**: Only 12 synthetic policy clauses
2. **No Context Understanding**: Semantic search may miss nuanced queries
3. **Single Document Retrieval**: Returns only the most relevant clause
4. **No Multi-hop Reasoning**: Can't combine information across clauses
5. **Embedding Model Limitations**: all-MiniLM-L6-v2 has limited domain knowledge
6. **TF-IDF Fallback**: Less accurate when sentence transformers unavailable

**Generation Limitations:**
1. **Rule-Based Explanations**: Not true natural language generation
2. **No LLM**: Doesn't synthesize or reason over information
3. **Template Responses**: Limited explanation variety
4. **No Summarization**: Returns full clause text
5. **No Conversation**: Doesn't maintain context or follow-ups
6. **Out-of-Scope Detection**: Simple keyword matching, may miss edge cases

**Content Limitations:**
1. **Synthetic Policy Clauses**: Not real insurance policies
2. **Simplified Content**: Actual policies are far more complex
3. **Static Knowledge**: Can't update or learn from interactions
4. **Limited Coverage**: Only 12 clauses, not comprehensive
5. **No Versioning**: Doesn't track policy versions or updates
6. **No Real-World Validation**: Clauses are fabricated for demonstration

### Potential Risks

**Misinformation Risk:**
- Users might mistake synthetic clauses for real policy language
- Simplified explanations may omit critical nuances
- Semantic search may retrieve irrelevant clauses for ambiguous queries

**Over-reliance Risk:**
- Users might use for actual policy interpretation
- May substitute for professional legal counsel
- Could lead to compliance violations if misused

**Technical Risks:**
- Poor retrieval quality with out-of-domain queries
- Missing relevant information due to semantic mismatch
- No validation of answer accuracy or applicability

### Recommendations

Users should:

- Understand this is a **demonstration only**
- Never use for actual compliance or legal decisions
- **Always consult qualified professionals** for policy interpretation
- Recognize the difference between this tool and production systems
- Review similarity scores critically (low scores indicate poor matches)
- Validate all outputs with compliance experts
- Understand that synthetic clauses don't reflect real policies

## How to Get Started with the Model

```python
import gradio as gr
from app import demo

# Launch the interface
demo.launch()
```

Or visit the Hugging Face Space to use the interactive demo.

## Training Details

Not applicable - this system uses pre-trained sentence transformers (all-MiniLM-L6-v2) for embedding and rule-based logic for explanations. No custom training was performed.

## Evaluation

### Testing Data, Factors & Metrics

The tool has been tested with various policy questions to ensure:

- Correct clause retrieval based on semantic similarity
- Appropriate similarity scoring
- Proper out-of-scope detection for pricing/underwriting questions
- Consistent human-in-the-loop warnings
- Graceful fallback to TF-IDF when sentence transformers unavailable

### Results

The system correctly:

- Loads 12 policy clauses from policy_clauses_snippets.txt
- Embeds clauses using sentence transformers or TF-IDF
- Retrieves most relevant clause based on cosine similarity
- Provides similarity scores (0.0-1.0 range)
- Generates explanations based on similarity thresholds:
  - High (>0.6): Directly related
  - Moderate (0.3-0.6): May contain relevant information
  - Low (<0.3): May not directly answer question
- Detects out-of-scope questions (pricing, legal advice, etc.)
- Enforces mandatory human review warnings

## Technical Specifications

### Model Architecture and Objective

**Architecture**: RAG pipeline with semantic retrieval

**Components:**
1. **Document Loader**: Loads policy clauses from text file
2. **Embedding Model**: Sentence transformers (all-MiniLM-L6-v2) or TF-IDF
3. **Retrieval Engine**: Cosine similarity search
4. **Explanation Generator**: Rule-based explanation based on similarity
5. **Warning System**: Mandatory human review enforcement

**Logic Flow:**
1. Load policy_clauses_snippets.txt and parse into 12 clauses
2. Embed all clauses using sentence transformers or TF-IDF
3. Receive user question
4. Check for out-of-scope keywords (pricing, legal advice, etc.)
5. Embed question using same method
6. Calculate cosine similarity between question and all clauses
7. Return clause with highest similarity
8. Generate explanation based on similarity score
9. Generate appropriate warning based on match quality
10. Display results with mandatory human review notice

### Compute Infrastructure

Runs on standard CPU infrastructure. Sentence transformers model is lightweight (~80MB). No GPU required.

## Environmental Impact

Minimal - uses small pre-trained model with negligible computational requirements. TF-IDF fallback has even lower resource usage.

## Citation

**BibTeX:**

```bibtex
@software{doc_rag_compliance_assistant,
  author = {Qoder for Vercept},
  title = {Document RAG Compliance Assistant},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/spaces/}}
}
```

## Glossary

- **RAG**: Retrieval-Augmented Generation - combining document retrieval with answer generation
- **Semantic Search**: Finding documents based on meaning, not just keywords
- **Embedding**: Vector representation of text that captures semantic meaning
- **Cosine Similarity**: Measure of similarity between two vectors (0=unrelated, 1=identical)
- **TF-IDF**: Term Frequency-Inverse Document Frequency - statistical measure of word importance
- **Out-of-Scope**: Questions beyond the system's intended use case

## More Information

For questions or feedback, please visit the Hugging Face Space discussion board.

## Model Card Authors

Qoder for Vercept

## Model Card Contact

Via Hugging Face Space discussions
