#!/usr/bin/env python3
"""
FORENSIC VALIDATION MASTER SCRIPT

Runs all Phase A gates in sequence:
- A0: Freeze the evidence
- A1: Environment sanity checks
- A2: MPPI verification

Stops at first failure.
"""

import os
import json
import subprocess
import sys

def run_gate(script_name, gate_name):
    """Run a gate script and return True if pass."""
    print("\n" + "="*70)
    print(f"RUNNING: {gate_name}")
    print("="*70)
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=True,
        text=True,
        cwd='/home/ulluboz/pcp-jepa-research'
    )
    
    print(result.stdout)
    
    if result.returncode != 0:
        print(f"ERROR OUTPUT:\n{result.stderr}")
    
    # Check results file
    results_file = f'/home/ulluboz/pcp-jepa-research/forensics/{gate_name.replace(" ", "_").lower()}_results.json'
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
            passed = results.get('pass', False)
    else:
        passed = result.returncode == 0
    
    return passed

def main():
    print("="*70)
    print("PHASE A: FORENSIC VALIDATION")
    print("="*70)
    print("""
Purpose: Prove our results are not bugs.

Gates:
  A0 - Freeze the evidence (reproducibility)
  A1 - Environment sanity (task is solvable)
  A2 - MPPI verification (planner works on known tasks)

If any gate fails, STOP and fix before proceeding.
""")
    
    # Track all gates
    all_gates = {
        'A0': False,
        'A1': False,
        'A2': False,
    }
    
    # Gate A0
    print("\n" + "="*70)
    print("GATE A0: FREEZE THE EVIDENCE")
    print("="*70)
    
    result = subprocess.run(
        [sys.executable, 'experiments/forensics/gate_A0_freeze_evidence.py'],
        capture_output=True,
        text=True,
        cwd='/home/ulluboz/pcp-jepa-research'
    )
    print(result.stdout)
    
    all_gates['A0'] = result.returncode == 0
    
    if not all_gates['A0']:
        print("\n❌ GATE A0 FAILED: Cannot freeze evidence")
        print("STOP: Fix reproducibility issues")
        return False
    
    # Gate A1
    print("\n" + "="*70)
    print("GATE A1: ENVIRONMENT SANITY CHECKS")
    print("="*70)
    
    result = subprocess.run(
        [sys.executable, 'experiments/forensics/gate_A1_environment_sanity.py'],
        capture_output=True,
        text=True,
        cwd='/home/ulluboz/pcp-jepa-research'
    )
    print(result.stdout)
    
    # Check results
    results_file = '/home/ulluboz/pcp-jepa-research/forensics/gate_A1_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
            all_gates['A1'] = results.get('pass', False)
    else:
        all_gates['A1'] = False
    
    if not all_gates['A1']:
        print("\n❌ GATE A1 FAILED: Environment has issues")
        print("STOP: Fix environment/task definition")
        return False
    
    # Gate A2
    print("\n" + "="*70)
    print("GATE A2: MPPI VERIFICATION")
    print("="*70)
    
    result = subprocess.run(
        [sys.executable, 'experiments/forensics/gate_A2_mppi_verification.py'],
        capture_output=True,
        text=True,
        cwd='/home/ulluboz/pcp-jepa-research'
    )
    print(result.stdout)
    
    # Check results
    results_file = '/home/ulluboz/pcp-jepa-research/forensics/gate_A2_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
            all_gates['A2'] = results.get('pass', False)
    else:
        all_gates['A2'] = False
    
    if not all_gates['A2']:
        print("\n❌ GATE A2 FAILED: MPPI implementation has issues")
        print("STOP: Fix MPPI before trusting results")
        return False
    
    # All gates passed
    print("\n" + "="*70)
    print("PHASE A COMPLETE")
    print("="*70)
    print(f"""
Gate Status:
  A0 (Evidence): {'✅ PASS' if all_gates['A0'] else '❌ FAIL'}
  A1 (Environment): {'✅ PASS' if all_gates['A1'] else '❌ FAIL'}
  A2 (MPPI): {'✅ PASS' if all_gates['A2'] else '❌ FAIL'}

{'✅ ALL GATES PASSED' if all(all_gates.values()) else '❌ SOME GATES FAILED'}

{'Ready to proceed to Phase B (Model validation).' if all(all_gates.values()) else 'Fix issues before proceeding.'}
""")
    
    # Save final status
    with open('/home/ulluboz/pcp-jepa-research/forensics/phase_A_status.json', 'w') as f:
        json.dump({
            'all_passed': all(all_gates.values()),
            'gates': all_gates
        }, f, indent=2)
    
    return all(all_gates.values())

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)