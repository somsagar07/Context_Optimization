"""
Test script for all workflow implementations.
Tests each workflow with a sample question and verifies execution.
"""
import sys
sys.path.append('.')

from agents_system import LLMWorker
from agents_system.workflows import (
    DirectWorkflow,
    PromptChainingWorkflow,
    RoutingWorkflow,
    ParallelSectioningWorkflow,
    ParallelVotingWorkflow,
    OrchestratorWorkersWorkflow,
    EvaluatorOptimizerWorkflow,
    AutonomousAgentWorkflow,
    get_workflow
)
from tools import ToolRegistry


def test_workflow(workflow_class, workflow_name, question, use_verifier=False):
    """Test a single workflow."""
    print(f"\n{'='*60}")
    print(f"Testing: {workflow_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize components
        worker = LLMWorker()
        tools = ToolRegistry()
        
        # Create workflow instance
        if workflow_class == PromptChainingWorkflow:
            workflow = workflow_class(worker, tools)
            workflow.use_verifier = use_verifier
        else:
            workflow = workflow_class(worker, tools)
        
        # Test parameters
        reasoner_tools = ["calculator"]
        reasoner_budget = 1  # Mid
        verifier_tools = []
        verifier_budget = 1  # Mid
        answerer_budget = 1  # Mid
        
        # Token budgets
        TOKEN_BUDGETS = {
            "reasoner": {0: 256, 1: 512, 2: 1024},
            "verifier": {0: 128, 1: 256, 2: 512},
            "answerer": {0: 64, 1: 128, 2: 256}
        }
        
        reasoner_tokens = TOKEN_BUDGETS["reasoner"][reasoner_budget]
        verifier_tokens = TOKEN_BUDGETS["verifier"][verifier_budget]
        answerer_tokens = TOKEN_BUDGETS["answerer"][answerer_budget]
        
        # Execute workflow
        print(f"Question: {question}")
        print(f"Executing workflow...")
        
        final_text, exec_info = workflow.execute(
            question,
            reasoner_tools,
            reasoner_budget,
            verifier_tools,
            verifier_budget,
            answerer_budget,
            reasoner_tokens,
            verifier_tokens,
            answerer_tokens
        )
        
        # Print results
        print(f"\n✓ Workflow executed successfully!")
        print(f"  Steps: {exec_info['steps']}")
        print(f"  Tools used: {exec_info['tools_count']}")
        print(f"  Total tokens: {exec_info['total_tokens']}")
        print(f"\nFinal Answer (first 200 chars):")
        print(f"  {final_text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow_registry():
    """Test the workflow registry."""
    print(f"\n{'='*60}")
    print(f"Testing Workflow Registry")
    print(f"{'='*60}")
    
    try:
        worker = LLMWorker()
        tools = ToolRegistry()
        
        for workflow_depth in range(9):
            workflow = get_workflow(workflow_depth, worker, tools)
            workflow_name = workflow.__class__.__name__
            print(f"  Workflow {workflow_depth}: {workflow_name} ✓")
        
        print(f"\n✓ All workflows registered correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        return False


def main():
    """Run all workflow tests."""
    print("="*60)
    print("WORKFLOW TEST SUITE")
    print("="*60)
    
    # Test question
    test_question = "What is 15 + 27?"
    
    # Test results
    results = []
    
    # Test registry
    results.append(("Registry", test_workflow_registry()))
    
    # Test individual workflows
    results.append(("Direct", test_workflow(DirectWorkflow, "Direct", test_question)))
    results.append(("Reason+Ans", test_workflow(PromptChainingWorkflow, "Reason+Ans", test_question, use_verifier=False)))
    results.append(("Reason+Verify+Ans", test_workflow(PromptChainingWorkflow, "Reason+Verify+Ans", test_question, use_verifier=True)))
    results.append(("Routing", test_workflow(RoutingWorkflow, "Routing", test_question)))
    results.append(("Parallel-Sectioning", test_workflow(ParallelSectioningWorkflow, "Parallel-Sectioning", test_question)))
    results.append(("Parallel-Voting", test_workflow(ParallelVotingWorkflow, "Parallel-Voting", test_question)))
    results.append(("Orchestrator-Workers", test_workflow(OrchestratorWorkersWorkflow, "Orchestrator-Workers", test_question)))
    results.append(("Evaluator-Optimizer", test_workflow(EvaluatorOptimizerWorkflow, "Evaluator-Optimizer", test_question)))
    results.append(("Autonomous-Agent", test_workflow(AutonomousAgentWorkflow, "Autonomous-Agent", test_question)))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

