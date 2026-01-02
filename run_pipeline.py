"""
Main entry point for running experiments.
"""
import argparse
from experiments.exp1_characterization import run_all_exp1
# from experiments.exp2_mechanisms import run_all_exp2
# from experiments.exp3_steering import run_all_exp3


def main():
    parser = argparse.ArgumentParser(description="Run reasoning engagement experiments")
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['exp1', 'exp2', 'exp3', 'all'],
        default='exp1',
        help='Which experiment to run'
    )
    
    args = parser.parse_args()
    
    if args.experiment == 'exp1' or args.experiment == 'all':
        print("Starting Experiment 1: Characterization")
        run_all_exp1()
    
    if args.experiment == 'exp2' or args.experiment == 'all':
        print("\nStarting Experiment 2: Mechanisms")
        # run_all_exp2()
        print("Experiment 2 not yet implemented")
    
    if args.experiment == 'exp3' or args.experiment == 'all':
        print("\nStarting Experiment 3: Steering")
        # run_all_exp3()
        print("Experiment 3 not yet implemented")


if __name__ == "__main__":
    main()
