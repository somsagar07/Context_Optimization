"""
Test script for all workflow implementations.
Tests each workflow with a sample question and verifies execution.
Supports both HuggingFace (local) and OpenRouter (API) workflows.
"""
import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents_system import LLMWorker, OpenRouterWorker
from agents_system.workflows import (
    # HuggingFace workflows
    DirectWorkflow,
    PromptChainingWorkflow,
    RoutingWorkflow,
    ParallelSectioningWorkflow,
    ParallelVotingWorkflow,
    OrchestratorWorkersWorkflow,
    EvaluatorOptimizerWorkflow,
    AutonomousAgentWorkflow,
    get_workflow,
    # OpenRouter workflows
    OpenRouterDirectWorkflow,
    OpenRouterPromptChainingWorkflow,
    OpenRouterRoutingWorkflow,
    OpenRouterParallelSectioningWorkflow,
    OpenRouterParallelVotingWorkflow,
    OpenRouterOrchestratorWorkersWorkflow,
    OpenRouterEvaluatorOptimizerWorkflow,
    OpenRouterAutonomousAgentWorkflow,
    get_openrouter_workflow,
)
from tools import ToolRegistry


def test_workflow(workflow_class, workflow_name, question, worker, use_verifier=False):
    """Test a single workflow."""
    print(f"\n{'='*60}")
    print(f"Testing: {workflow_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize components
        tools = ToolRegistry()
        
        # Create workflow instance
        if workflow_class in [PromptChainingWorkflow, OpenRouterPromptChainingWorkflow]:
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


def test_workflow_registry(worker, get_workflow_func, worker_type):
    """Test the workflow registry."""
    print(f"\n{'='*60}")
    print(f"Testing {worker_type} Workflow Registry")
    print(f"{'='*60}")
    
    try:
        tools = ToolRegistry()
        
        for workflow_depth in range(9):
            workflow = get_workflow_func(workflow_depth, worker, tools)
            workflow_name = workflow.__class__.__name__
            print(f"  Workflow {workflow_depth}: {workflow_name} ✓")
        
        print(f"\n✓ All {worker_type} workflows registered correctly!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_huggingface_workflows(question):
    """Test all HuggingFace workflows."""
    print("="*60)
    print("HUGGINGFACE WORKFLOW TEST SUITE")
    print("="*60)
    
    results = []
    
    try:
        worker = LLMWorker()
        tools = ToolRegistry()
        
        # Test registry
        results.append(("Registry", test_workflow_registry(worker, get_workflow, "HuggingFace")))
        
        # Test individual workflows
        results.append(("Direct", test_workflow(DirectWorkflow, "Direct", question, worker)))
        results.append(("Reason+Ans", test_workflow(PromptChainingWorkflow, "Reason+Ans", question, worker, use_verifier=False)))
        results.append(("Reason+Verify+Ans", test_workflow(PromptChainingWorkflow, "Reason+Verify+Ans", question, worker, use_verifier=True)))
        results.append(("Routing", test_workflow(RoutingWorkflow, "Routing", question, worker)))
        results.append(("Parallel-Sectioning", test_workflow(ParallelSectioningWorkflow, "Parallel-Sectioning", question, worker)))
        results.append(("Parallel-Voting", test_workflow(ParallelVotingWorkflow, "Parallel-Voting", question, worker)))
        results.append(("Orchestrator-Workers", test_workflow(OrchestratorWorkersWorkflow, "Orchestrator-Workers", question, worker)))
        results.append(("Evaluator-Optimizer", test_workflow(EvaluatorOptimizerWorkflow, "Evaluator-Optimizer", question, worker)))
        results.append(("Autonomous-Agent", test_workflow(AutonomousAgentWorkflow, "Autonomous-Agent", question, worker)))
        
    except Exception as e:
        print(f"\n✗ Error initializing HuggingFace worker: {str(e)}")
        import traceback
        traceback.print_exc()
        return []
    
    return results


def test_openrouter_workflows(question):
    """Test all OpenRouter workflows."""
    print("="*60)
    print("OPENROUTER WORKFLOW TEST SUITE")
    print("="*60)
    
    results = []
    
    try:
        worker = OpenRouterWorker()
        tools = ToolRegistry()
        
        # Test registry
        results.append(("Registry", test_workflow_registry(worker, get_openrouter_workflow, "OpenRouter")))
        
        # Test individual workflows
        results.append(("Direct", test_workflow(OpenRouterDirectWorkflow, "Direct (API)", question, worker)))
        results.append(("Reason+Ans", test_workflow(OpenRouterPromptChainingWorkflow, "Reason+Ans (API)", question, worker, use_verifier=False)))
        results.append(("Reason+Verify+Ans", test_workflow(OpenRouterPromptChainingWorkflow, "Reason+Verify+Ans (API)", question, worker, use_verifier=True)))
        results.append(("Routing", test_workflow(OpenRouterRoutingWorkflow, "Routing (API)", question, worker)))
        results.append(("Parallel-Sectioning", test_workflow(OpenRouterParallelSectioningWorkflow, "Parallel-Sectioning (API)", question, worker)))
        results.append(("Parallel-Voting", test_workflow(OpenRouterParallelVotingWorkflow, "Parallel-Voting (API)", question, worker)))
        results.append(("Orchestrator-Workers", test_workflow(OpenRouterOrchestratorWorkersWorkflow, "Orchestrator-Workers (API)", question, worker)))
        results.append(("Evaluator-Optimizer", test_workflow(OpenRouterEvaluatorOptimizerWorkflow, "Evaluator-Optimizer (API)", question, worker)))
        results.append(("Autonomous-Agent", test_workflow(OpenRouterAutonomousAgentWorkflow, "Autonomous-Agent (API)", question, worker)))
        
    except Exception as e:
        print(f"\n✗ Error initializing OpenRouter worker: {str(e)}")
        print("  Make sure OPENROUTER_API_KEY is set in .env file or environment variable")
        import traceback
        traceback.print_exc()
        return []
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test workflow implementations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test HuggingFace workflows
  python agents_system/test_workflows.py --workflow huggingface
  
  # Test OpenRouter workflows
  python agents_system/test_workflows.py --workflow api
  
  # Test both (default)
  python agents_system/test_workflows.py
        """
    )
    
    parser.add_argument(
        "--workflow",
        type=str,
        choices=["huggingface", "api", "both"],
        default="both",
        help="Which workflows to test: 'huggingface' (local models), 'api' (OpenRouter), or 'both' (default)"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        default="What is 15 + 27?",
        help="Test question to use (default: 'What is 15 + 27?')"
    )
    
    return parser.parse_args()


def main():
    """Run workflow tests."""
    args = parse_args()
    
    test_question = args.question
    
    all_results = []
    
    # Test HuggingFace workflows
    if args.workflow in ["huggingface", "both"]:
        hf_results = test_huggingface_workflows(test_question)
        all_results.extend([(f"HF-{name}", result) for name, result in hf_results])
    
    # Test OpenRouter workflows
    if args.workflow in ["api", "both"]:
        api_results = test_openrouter_workflows(test_question)
        all_results.extend([(f"API-{name}", result) for name, result in api_results])
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    
    for name, result in all_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:30s} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

