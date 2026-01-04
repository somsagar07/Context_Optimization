# Workflows

## Overview

The system supports 9 distinct workflows for agentic reasoning, each with different patterns and agent configurations.

## Workflow 0: Direct

**Pattern**: Single-step direct answer

```
Question → [Answerer with agent1_tools] → Final Answer
```

**Agents**: Answerer (uses `agent1_tools`)

**Tool Assignment**: 
- Agent1 Tools: Answerer
- Agent2 Tools: Not used

---

## Workflow 1: Reason+Ans

**Pattern**: Reason then answer

```
Question → [Reasoner with agent1_tools] → Reasoning → [Answerer] → Final Answer
```

**Agents**: Reasoner (agent1_tools) → Answerer

**Tool Assignment**:
- Agent1 Tools: Reasoner
- Agent2 Tools: Not used

---

## Workflow 2: Reason+Verify+Ans

**Pattern**: Reason, verify, then answer

```
Question → [Reasoner with agent1_tools] → Reasoning → [Verifier with agent2_tools] → Critique → [Answerer] → Final Answer
```

**Agents**: Reasoner (agent1_tools) → Verifier (agent2_tools) → Answerer

**Tool Assignment**:
- Agent1 Tools: Reasoner
- Agent2 Tools: Verifier

---

## Workflow 3: Routing

**Pattern**: Classify → Route to one of two reasoners → Answer

```
Question → [Classifier with agent1_tools] → Classification
         ↓
    Route Decision:
    - Simple/Complex → [Reasoner1 with agent1_tools]
    - Multi-step → [Reasoner2 with agent2_tools]
         ↓
    [Answerer] → Final Answer
```

**Agents**: 
- Classifier (agent1_tools)
- Routing Logic: 
  - If classification contains "multi-step" → Route to Reasoner2 (agent2_tools)
  - Otherwise → Route to Reasoner1 (agent1_tools)
- Answerer

**Tool Assignment**:
- Agent1 Tools: Classifier, Reasoner1 (if routed)
- Agent2 Tools: Reasoner2 (if routed)

---

## Workflow 4: Parallel-Sectioning

**Pattern**: Break down → Two workers (one agent1, one agent2) → Combine

```
Question → [Aspect1/Breakdown Agent with agent1_tools] → Task Breakdown (2 subtasks)
         ↓
    [Worker1 with agent2_tools] → Subtask 1 Result
    [Worker2 with agent1_tools] → Subtask 2 Result (reuses agent1)
         ↓
    [Answerer] → Final Answer (combines both)
```

**Agents**:
- Aspect1/Breakdown Agent (agent1_tools) - breaks down into 2 subtasks
- Worker1 (agent2_tools) - solves subtask 1
- Worker2 (agent1_tools, reused) - solves subtask 2
- Answerer

**Tool Assignment**:
- Agent1 Tools: Breakdown Agent, Worker2
- Agent2 Tools: Worker1

**Key Feature**: Breakdown agent also works as a worker (Worker2 reuses agent1)

---

## Workflow 5: Parallel-Voting

**Pattern**: Multiple parallel attempts → Aggregate

```
Question → [Vote 1 with agent1_tools] → Answer 1
         → [Vote 2 with agent1_tools] → Answer 2
         → [Vote 3 with agent1_tools] → Answer 3
         ↓
    [Answerer] → Final Answer (aggregates votes)
```

**Agents**: Vote Agents (3x, agent1_tools) → Answerer

**Tool Assignment**:
- Agent1 Tools: Votes (3x)
- Agent2 Tools: Not used

---

## Workflow 6: Orchestrator-Workers

**Pattern**: Orchestrator breaks down → Multiple workers → Synthesize

```
Question → [Orchestrator with agent1_tools] → Task Breakdown
         ↓
    [Worker1 with agent2_tools] → Subtask 1 Result
    [Worker2 with agent2_tools] → Subtask 2 Result
         ↓
    [Answerer] → Final Answer (synthesizes)
```

**Agents**:
- Orchestrator (agent1_tools) - only breaks down, doesn't work
- Workers (2x, agent2_tools) - all use same tools
- Answerer

**Tool Assignment**:
- Agent1 Tools: Orchestrator
- Agent2 Tools: Workers (2x)

**Difference from Workflow 4**:
- **Workflow 4**: Breakdown agent also works (Worker2 reuses agent1)
- **Workflow 6**: Orchestrator is separate (pure delegation), all workers use agent2_tools

---

## Workflow 7: Evaluator-Optimizer

**Pattern**: Generate → Evaluate → Refine loop

```
Question → [Generator with agent1_tools] → Answer
         ↓
    [Evaluator with agent2_tools] → Critique
         ↓
    [Generator with agent1_tools] → Refined Answer (if needed)
         ↓
    [Evaluator with agent2_tools] → Critique
         ↓
    [Answerer] → Final Answer
```

**Agents**:
- Generator (agent1_tools) - generates/refines
- Evaluator (agent2_tools) - evaluates
- Loop: Generate → Evaluate → (if not good) Refine → Evaluate → ...
- Answerer

**Tool Assignment**:
- Agent1 Tools: Generator
- Agent2 Tools: Evaluator

**Key Feature**: Iterative refinement with explicit evaluation

---

## Workflow 8: Autonomous-Agent

**Pattern**: Autonomous iterative reasoning

```
Question → [Iteration 1 with agent1_tools] → Reasoning 1
         → [Iteration 2 with agent2_tools] → Reasoning 2
         → [Iteration 3 with agent2_tools] → Reasoning 3
         ↓
    [Answerer] → Final Answer
```

**Agents**:
- Iteration 1 (agent1_tools)
- Iterations 2+ (agent2_tools)
- Answerer

**Tool Assignment**:
- Agent1 Tools: Iteration 1
- Agent2 Tools: Iterations 2+

**Key Feature**: Autonomous iterative reasoning without explicit evaluation

---

## Workflow Comparison

| Workflow | Pattern | Key Feature | Uses Agent2? |
|----------|---------|-------------|--------------|
| 0 | Direct | Single agent | No |
| 1 | Chain | Reason → Answer | No |
| 2 | Chain+Verify | Reason → Verify → Answer | Yes (Verifier) |
| 3 | Routing | Classify → Route to Reasoner1 OR Reasoner2 | Yes (Reasoner2) |
| 4 | Parallel-Breakdown | Breakdown agent also works | Yes (Worker1) |
| 5 | Parallel-Voting | Multiple votes → Aggregate | No |
| 6 | Orchestrator | Pure delegation (orchestrator doesn't work) | Yes (Workers) |
| 7 | Iterative-Eval | Generate → Evaluate → Refine loop | Yes (Evaluator) |
| 8 | Autonomous | Iterative reasoning without evaluation | Yes (Iterations 2+) |

**All workflows are distinct and serve different purposes!**

---

## Tool Assignment Summary

| Workflow | Agent1 Tools Used By | Agent2 Tools Used By |
|----------|---------------------|---------------------|
| 0 | Answerer | - |
| 1 | Reasoner | - |
| 2 | Reasoner | Verifier |
| 3 | Classifier, Reasoner1 (if routed) | Reasoner2 (if routed) |
| 4 | Breakdown Agent, Worker2 | Worker1 |
| 5 | Votes (3x) | - |
| 6 | Orchestrator | Workers (2x) |
| 7 | Generator | Evaluator |
| 8 | Iteration 1 | Iterations 2+ |

---

## Answerer Tool Usage

- **Workflow 0**: Answerer uses `agent1_tools` (it's the only agent)
- **Workflows 1-8**: Answerer uses no tools (just synthesizes from context provided by previous agents)

