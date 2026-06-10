# Week 3: Recurrent Neural Networks (RNNs), Attention, and Sequence-to-Sequence Models

Welcome to **Week 3** of the Deep Learning Project Track!

This week focuses on sequence modeling using **Recurrent Neural Networks (RNNs)**, **Attention Mechanisms**, and **Sequence-to-Sequence (Seq2Seq)** architectures. These concepts form the foundation of modern Natural Language Processing (NLP) and are important stepping stones toward understanding Transformer-based models. This is foundational in upcoming weeks.

---

## Learning Objectives

By the end of this week, you should be able to:

- Understand how Recurrent Neural Networks process sequential data.
- Implement RNNs using PyTorch's `torch.nn` modules.
- Understand hidden states and how information flows through sequences.
- Learn the motivation behind Attention mechanisms.
- Build and train encoder-decoder architectures.
- Understand Long Short-Term Memory (LSTM) networks.
- Implement sequence-to-sequence learning models.
- Analyze tensor shapes and data flow in recurrent architectures.

---

## Repository Structure

```text
Week3/
│
├── task_1.ipynb
├── task_2a.ipynb
├── task_2b.ipynb
├── task_3.ipynb
├── LSTM_Assignment.ipynb
│
├── reference_materials/
│   ├── attention_rnn_paper.pdf
│   ├── LSTMs.pdf
│   ├── RNNs.pdf
│   ├──EncoderDecoder.pdf
│
└── bonus/
    └── problem_statement.pdf
```

---

# Tasks

## Task 1 — Cumulative Sum Prediction using RNNs

Build a recurrent neural network that learns cumulative sums over sequences.

### Concepts Covered

- Embedding layers
- Recurrent Neural Networks
- Hidden states
- Sequence prediction
- PyTorch RNN modules

### Goal

Learn how information propagates through a sequence and how recurrent models retain context from previous timesteps.

---

## Task 2A — Token-wise Function Learning

Train an RNN to learn a simple transformation:

```text
target = 2 × input + K
```

### Concepts Covered

- Sequence processing
- Classification over sequence positions
- Vocabulary construction
- Learning deterministic token transformations

---

## Task 2B — Local Window Sum Prediction

Predict the sum of values around each position in a sequence.

### Concepts Covered

- Context aggregation
- Local sequence reasoning
- Learning dependencies between neighboring tokens

---

## Task 3 — Attention-Based Sequence Modeling

Build a model that uses an encoder and an attention mechanism to solve a sequence task.

### Concepts Covered

- Encoder representations
- Attention scores
- Context vectors
- Alignment between input and output sequences

### Goal

Understand why attention mechanisms became necessary and how they improve upon vanilla recurrent models.

---

## LSTM Seq2Seq Assignment

This notebook consists of two parts.

### Part A — Addition as a Mapping Problem

Train an LSTM to directly learn:

```text
[a, b] → a + b
```

### Concepts Covered

- LSTM networks
- Sequence representation
- Regression / classification mapping

---

### Part B — Addition as a Sequence-to-Sequence Problem

Build an Encoder-Decoder architecture that performs addition as a character-level sequence task.

Example:

```text
Input : "123+456"
Output: "579"
```

### Concepts Covered

- Encoder-Decoder LSTMs
- Sequence generation
- Teacher Forcing
- Character tokenization
- Seq2Seq learning

---

# Submission Guidelines

1. Complete all TODO sections.
2. Do not modify the provided evaluation code unless instructed.
3. Ensure every notebook runs from start to finish without errors.
4. Save notebook outputs before submission.
5. Submit only the required notebook files.
6. Clearly document any assumptions or design decisions.

---

# Resources

## Official PyTorch Documentation

These assignments are intended to strengthen your understanding of PyTorch's neural network modules.

### Core Documentation

- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/)
- [torch.nn Documentation](https://docs.pytorch.org/docs/stable/nn.html)
- [torch.nn.RNN](https://docs.pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [torch.nn.LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

---

## Required Reading

### Attention Mechanisms

**File:** `reference_materials/attention_rnn_paper.pdf`

This paper introduces one of the most influential ideas in deep learning:

- Neural Machine Translation
- Attention Mechanisms
- Context vectors
- Encoder-Decoder architectures

Students are encouraged to at least skim the paper before attempting Task 3.

---

# Additional Learning Resources

The following resources are optional but highly recommended.

## Harvard CS109B Notebook

https://harvard-iacs.github.io/2021-CS109B/lectures/lecture21/notebook/

Topics covered:

- Recurrent Neural Networks
- Backpropagation Through Time (BPTT)
- Long Short-Term Memory Networks (LSTMs)
- Sequence Modeling
- Language Modeling
- Encoder-Decoder Architectures

This notebook provides a practical and intuitive walkthrough of the concepts used throughout this week's assignments.

---

## Harvard LSTM Lecture Slides

**File:** `reference_materials/LSTMs.pdf`

Topics covered:

- RNN training dynamics
- Hidden state propagation
- Backpropagation Through Time (BPTT)
- Sequence generation
- Long Short-Term Memory (LSTM) Networks
- Practical intuition behind recurrent architectures

---

# Bonus Challenge

Located in:

```text
bonus/problem_statement.pdf
```

Attempt this only after completing all core tasks.

The bonus challenge is designed for students who want to explore sequence modeling beyond the assignment requirements.

---

# Suggested Exploration Topics

If you finish early and want to dive deeper:

### Recurrent Models

- Vanilla RNNs
- Deep RNNs
- Bidirectional RNNs

### LSTMs

- Forget Gates
- Input Gates
- Output Gates
- Memory Cells

### Attention

- Bahdanau Attention
- Luong Attention
- Self-Attention

### Seq2Seq

- Machine Translation
- Text Summarization
- Dialogue Systems

### Beyond RNNs

- Transformers
- BERT
- GPT-style Architectures

---

# Recommended Workflow

1. Read the task description carefully.
2. Review the relevant PyTorch documentation.
3. Understand the tensor dimensions before coding.
4. Train simple models first.
5. Visualize predictions and errors.
6. Debug using tensor shapes.
7. Attempt the bonus challenge after completing the mandatory tasks.

---

# Expected Outcomes

By the end of Week 3, you should be comfortable with:

1) Building RNNs in PyTorch

2) Understanding hidden states and sequence modeling

3) Training LSTM-based architectures

4) Implementing Attention Mechanisms

5) Building Encoder-Decoder Networks

6) Solving Sequence-to-Sequence Tasks

7) Reading and understanding research papers

---
