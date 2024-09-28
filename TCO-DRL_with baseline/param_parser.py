import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="SAIRL")

    # General
    parser.add_argument("--Baselines",
                        type=list,
                        default=['Random', 'Round-Robin', 'Earliest', 'DQN', 'BLOR', 'SemiGreedy'],
                        help="Experiment Baseline")
    parser.add_argument("--Baseline_num",
                        type=int,
                        default=0,
                        help="Number of baselines")


    parser.add_argument("--Epoch",
                        type=int,
                        default=5,
                        help="Training Epochs")

    # DQN
    parser.add_argument("--Dqn_start_learn",
                        type=int,
                        default=500,
                        help="Iteration start Learn for normal dqn")
    parser.add_argument("--Dqn_learn_interval",
                        type=int,
                        default=1,
                        help="Dqn's learning interval")

    # Oracle Settings
    parser.add_argument("--Oracle_Type",
                        type=list,
                        default=[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                        help="Oracle Type")
    parser.add_argument("--Oracle_Cost",
                        type=list,
                        default=[0.3, 0.3, 0.6, 0.6, 0.9, 0.3, 0.3, 0.6, 0.6, 0.9, 0.3, 0.3, 0.6, 0.6, 0.9],
                        help="Oracle Cost")
    parser.add_argument("--Oracle_Acc",
                        type=list,
                        default=[1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2, 1, 1, 1.1, 1.1, 1.2],
                        help="Oracle Cpus")
    parser.add_argument("--Oracle_Tokens",
                        type=list,
                        default=[150, 150, 300, 300, 500, 150, 150, 300, 300, 500, 150, 150, 300, 300, 500],
                        help="Oracle Staked tokens")
    parser.add_argument("--Oracle_Behavior_Probs",
                        type=list,
                        default=[[0.5, 0.25, 0.2, 0.05], [0.9, 0.1, 0, 0], [0.65, 0.2, 0.15, 0], [0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0],
                                 [0.5, 0.25, 0.2, 0.05], [0.9, 0.1, 0, 0], [0.65, 0.2, 0.15, 0], [0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0],
                                 [0.5, 0.25, 0.2, 0.05], [0.9, 0.1, 0, 0], [0.65, 0.2, 0.15, 0], [0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0]],

                        help="Oracle Behavior Probs")
    parser.add_argument("--Oracle_Validation_Probs",
                        type=list,
                        default=[0.5, 0.8, 0.7, 0.85, 0.95, 0.5, 0.8, 0.7, 0.85, 0.95, 0.5, 0.8, 0.7, 0.85, 0.95],

                        help="Oracle Validation Probs")
    parser.add_argument("--Oracle_Num",
                        type=int,
                        default=15,
                        help="The number of Oracles")
    parser.add_argument("--Oracle_Initial_Reputation",
                        type=float,
                        default=0.5,
                        help="The initial reputation of Oracles")
    parser.add_argument("--Time_Window_Size",
                        type=int,
                        default=5,
                        help="The size of Time window")
    parser.add_argument("--Time_Period_Size",
                        type=int,
                        default=60,
                        help="The size of Time period")
    parser.add_argument("--Oracle_capacity",
                        type=int,
                        default=1000,
                        help="Oracle capacity")

    # Request Settings
    parser.add_argument("--lamda",
                        type=int,
                        default=5,
                        help="The parameter used to control the interval time of each requests.")
    parser.add_argument("--Request_Num",
                        type=int,
                        default=6000,
                        help="The number of requests.")
    parser.add_argument("--Request_len_Mean",
                        type=int,
                        default=6000,
                        help="The mean value of the normal distribution.")
    parser.add_argument("--Request_len_Std",
                        type=int,
                        default=500,
                        help="The std value of the normal distribution.")
    parser.add_argument("--Request_ddl",
                        type=float,
                        default=7.0,
                        help="Deadline time of each requests")
    return parser.parse_args()