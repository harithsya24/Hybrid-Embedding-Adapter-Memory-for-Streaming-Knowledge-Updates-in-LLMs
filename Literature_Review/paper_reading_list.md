# Continual Learning for LLMs: Week 1 Reading List
## Hybrid Memory System Project

---

## **Monday (1hr): Parameter-Efficient Fine-Tuning Papers (LoRA, Adapters)**

### Core Papers

1. **LoRA: Low-Rank Adaptation of Large Language Models** (arXiv 2106.09685)
   - Authors: Edward J. Hu et al.
   - Link: https://arxiv.org/abs/2106.09685
   - Summary: The foundational LoRA paper that introduces rank decomposition matrices to reduce trainable parameters by 10,000x while maintaining performance
   - Key contribution: Freezes pretrained weights and injects trainable low-rank matrices into each Transformer layer

2. **Parameter-Efficient Transfer Learning for NLP** (ICML 2019)
   - Authors: Houlsby et al.
   - Link: https://arxiv.org/pdf/1902.00751
   - Summary: Original adapter method paper - adds bottleneck layers after attention and FFN layers
   - Key contribution: Achieves near state-of-the-art with only 3% task-specific parameters

### Supplementary Papers

3. **Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey**
   - Link: https://arxiv.org/pdf/2403.14608
   - Summary: Comprehensive overview of PEFT methods including adapters, LoRA, prefix tuning, and hybrid approaches

4. **La-LoRA: Parameter-efficient fine-tuning with layer-wise adaptive low-rank adaptation**
   - Link: https://www.sciencedirect.com/science/article/abs/pii/S089360802500975X
   - Summary: Extends LoRA with dynamic rank allocation per layer based on contribution

### Practical Resources

- Hugging Face PEFT Library: https://github.com/huggingface/peft
- LoRA Documentation: https://huggingface.co/docs/peft/main/en/conceptual_guides/lora
- Keras LoRA Tutorial: https://keras.io/examples/keras_recipes/parameter_efficient_finetuning_of_gemma_with_lora_and_qlora/

---

## **Tuesday (1hr): Embedding-Based Continual Learning Papers**

### Core Papers

1. **Efficient continual learning in neural networks with embedding regularization** (Neural Networks 2020)
   - Link: https://www.sciencedirect.com/science/article/abs/pii/S092523122030151X
   - Summary: Proposes embedding regularization (ER) for continual learning - regularizes internal embeddings rather than outputs or weights
   - Key contribution: More efficient than EWC and GEM, allows network more freedom in enforcing penalties

2. **In-context Continual Learning Assisted by an External Continual Learner** (COLING 2025)
   - arXiv: https://arxiv.org/abs/2412.15563
   - Summary: InCA method uses embeddings of LLM-generated tags to represent classes as Gaussian distributions
   - Key contribution: Avoids catastrophic forgetting without fine-tuning, handles scalability issues

### Supplementary Papers

3. **LLMs are Also Effective Embedding Models: An In-depth Overview**
   - Link: https://arxiv.org/html/2412.12591v1
   - Summary: Comprehensive survey on using LLMs as embedding models, covers continual pre-training for domain adaptation

4. **Continual Pre-training of Language Models** (ICLR)
   - Link: https://openreview.net/pdf?id=m_GDIItaI3o
   - Summary: DAS method with soft-masking for continual domain-adaptive pre-training
   - Key contribution: Enables forward/backward transfer without domain IDs

### Resources

- GitHub Survey: https://github.com/Wang-ML-Lab/llm-continual-learning-survey
- Comprehensive CL Survey: https://github.com/zzz47zzz/awesome-lifelong-learning-methods-for-llm

---

## **Wednesday (1hr): Streaming/Online Learning for LLMs Papers**

### Core Papers

1. **Efficient Streaming Language Models with Attention Sinks** (ICLR 2024)
   - arXiv: https://arxiv.org/abs/2309.17453
   - Summary: StreamingLLM framework enables infinite sequence length by keeping initial tokens as "attention sinks"
   - Key contribution: Up to 22.2x speedup, handles 4M+ tokens without fine-tuning
   - Code: https://github.com/mit-han-lab/streaming-llm

2. **VideoLLM-online: Online Video Large Language Model for Streaming Video** (CVPR 2024)
   - arXiv: https://arxiv.org/abs/2406.11816
   - Website: https://showlab.github.io/videollm-online/
   - Summary: LIVE framework for temporally aligned, real-time conversation in continuous video streams
   - Key contribution: 5-10 FPS on RTX 3090, parallelized inference architecture
   - Code: https://github.com/showlab/videollm-online

### Supplementary Papers

3. **LiveCC: Learning Video LLM with Streaming Speech Transcription at Scale**
   - Link: https://showlab.github.io/livecc
   - Summary: Trains on millions of streaming ASR transcripts, enables real-time commentary
   - Key contribution: Learns from incomplete, streaming ASR words aligned with video frames

4. **Streaming Long Video Understanding with Large Language Models** (NeurIPS 2024)
   - Summary: Memory-Propagated Streaming Encoding with Adaptive Memory Selection
   - Key contribution: Fixed-length memory for arbitrarily long videos

---

## **Thursday (1hr): Hybrid Memory Systems Papers**

### Core Papers

1. **A-MEM: Agentic Memory for LLM Agents**
   - arXiv: https://arxiv.org/abs/2502.12110
   - Summary: Dynamic memory organization following Zettelkasten principles
   - Key contribution: Agent-driven memory indexing, linking, and evolution
   - Code: https://github.com/agiresearch/A-mem

2. **Memory in Large Language Models: Mechanisms, Evaluation and Evolution**
   - arXiv: https://arxiv.org/html/2509.18868v1
   - Summary: Unified framework covering parametric, contextual, external, and episodic memory
   - Key contribution: Comprehensive evaluation metrics and governance framework

### Supplementary Papers

3. **Memory in LLM-based Multi-agent Systems** (TechRxiv 2025)
   - Summary: Analysis of memory topologies (centralized, distributed, hybrid) in multi-agent systems
   - Key contribution: Trade-offs in scalability, robustness, and privacy

4. **Building AI Agents with Memory Systems: Cognitive Architectures for LLMs**
   - Link: https://bluetickconsultants.medium.com/building-ai-agents-with-memory-systems-cognitive-architectures-for-llms-176d17e642e7
   - Summary: Working, episodic, semantic, and procedural memory integration

### Resources

- Agent Memory Papers: https://github.com/Shichun-Liu/Agent-Memory-Paper-List
- ICLR 2026 Workshop: https://openreview.net/pdf?id=U51WxL382H

---

## **Friday (1hr): Knowledge Editing and Memory Augmentation Papers**

### Core Papers

1. **WISE: Rethinking the Knowledge Memory for Lifelong Model Editing** (NeurIPS 2024)
   - Link: https://openreview.net/pdf?id=VJMYOfJVC2
   - Summary: Dual parametric memory (main + side) with routing mechanism
   - Key contribution: Knowledge sharding and merging for continual editing without conflicts
   - NeurIPS Poster: https://neurips.cc/virtual/2024/poster/94912

2. **MAKE: Memory-Associated Knowledge Editing** (TACL 2025)
   - Link: https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.26/132652/
   - Summary: Leverages internal LLM knowledge for editing with associated knowledge transfer
   - Key contribution: Recalls indirect associated knowledge from the model itself

### Supplementary Papers

3. **Augmenting Language Models with Long-Term Memory** (NeurIPS 2023)
   - Link: https://arxiv.org/pdf/2306.07174
   - Summary: LONGMEM framework with decoupled memory via residual SideNet
   - Key contribution: Separates encoding and retrieval to avoid memory staleness

4. **How memory augmentation can improve large language models** (IBM Research)
   - Link: https://research.ibm.com/blog/memory-augmented-LLMs
   - Summary: Larimar system for one-shot gradient-free memory updates
   - Key contribution: Episodic memory controller for fast knowledge editing

### Resources

- Knowledge Editing Papers: https://github.com/zjunlp/KnowledgeEditingPapers
- Memory-Augmented LLM Overview: https://www.emergentmind.com/topics/memory-augmented-llm
- NACE.AI Tutorial: https://nace.ai/research/memory-augmentation-and-editing-techniques-in-large-language-models

---

## **Additional Resources for Weekend Implementation**

### Frameworks & Libraries

1. **Hugging Face PEFT**
   - GitHub: https://github.com/huggingface/peft
   - Documentation: Comprehensive PEFT methods implementation

2. **LLM Continual Learning Survey**
   - GitHub: https://github.com/Wang-ML-Lab/llm-continual-learning-survey
   - ACM Survey Paper: https://dl.acm.org/doi/10.1145/3735633

### Key Concepts to Track

**For Your Hybrid System:**

1. **Memory Components:**
   - Parametric memory (model weights) via LoRA
   - Working memory (embeddings) for continual learning
   - External memory (retrieval) for knowledge augmentation
   - Episodic memory for streaming data

2. **Key Techniques to Integrate:**
   - LoRA for efficient parameter updates
   - Embedding regularization for continual learning
   - Attention sinks for streaming
   - Dual memory architecture for knowledge editing
   - Adaptive memory selection and routing

3. **Evaluation Metrics:**
   - Reliability, generalization, locality (impossible triangle)
   - Catastrophic forgetting measurement
   - Forward/backward transfer
   - Memory efficiency and retrieval speed

---

## **Notes for Implementation**

### Saturday Setup Checklist:
- [ ] Python environment (3.9+)
- [ ] PyTorch 2.0+
- [ ] Hugging Face Transformers & PEFT
- [ ] Vector database (ChromaDB/FAISS) for embeddings
- [ ] GPU access (for experiments)

### Sunday Architecture Components:
- [ ] Base LLM with LoRA adapters
- [ ] Embedding storage and retrieval system
- [ ] Streaming data pipeline
- [ ] Memory routing mechanism
- [ ] Knowledge editing interface

### Key Design Decisions:
1. **Memory Hierarchy:** Main (pretrained) + Side (adapted) + External (retrieved)
2. **Update Strategy:** LoRA for side memory, embeddings for retrieval
3. **Streaming Handling:** Attention sinks + sliding window
4. **Routing:** Learned router to decide memory source

Good luck with your reading and implementation! This is a cutting-edge research area with lots of exciting opportunities.
