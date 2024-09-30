import sys
import json
import argparse
from tabulate import tabulate

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def get_stat(agent_stats, key):
    """
    Helper function to retrieve values from agent statistics, replacing missing fields with 0.

    """
    return agent_stats.get(key, 0)

def compare_agents(data, filename):
    agents = data['by_agent']
    own_agent_name = list(agents.keys())[0]
    own_agent = agents[own_agent_name]
    comparisons = []

    own_coins = get_stat(own_agent, 'coins')
    own_kills = get_stat(own_agent, 'kills')
    own_original_reward = own_coins * 1 + own_kills * 3

    for agent_name, stats in agents.items():
        if agent_name == own_agent_name:
            continue

        agent_coins = get_stat(stats, 'coins')
        agent_kills = get_stat(stats, 'kills')
        agent_original_reward = agent_coins * 1 + agent_kills * 3

        comparison = {
            'Metric': f"{own_agent_name} ({filename}) vs {agent_name}",
            'Original Reward': f"{own_original_reward} vs {agent_original_reward} ({own_original_reward - agent_original_reward:+})",
            'Score': f"{get_stat(own_agent, 'score')} vs {get_stat(stats, 'score')} ({get_stat(own_agent, 'score') - get_stat(stats, 'score'):+})",
            'Coins': f"{own_coins} vs {agent_coins} ({own_coins - agent_coins:+})",
            'Kills': f"{own_kills} vs {agent_kills} ({own_kills - agent_kills:+})",
            'Suicides': f"{get_stat(own_agent, 'suicides')} vs {get_stat(stats, 'suicides')} ({get_stat(own_agent, 'suicides') - get_stat(stats, 'suicides'):+})",
            'Invalid Actions': f"{get_stat(own_agent, 'invalid')} vs {get_stat(stats, 'invalid')} ({get_stat(own_agent, 'invalid') - get_stat(stats, 'invalid'):+})",
        }
        comparisons.append(comparison)
    return comparisons

def compare_own_agents(data1, data2, filename1, filename2):
    agents1 = data1['by_agent']
    agents2 = data2['by_agent']

    own_agent_name1 = list(agents1.keys())[0]
    own_agent_name2 = list(agents2.keys())[0]

    own_agent1 = agents1[own_agent_name1]
    own_agent2 = agents2[own_agent_name2]

    own_coins1 = get_stat(own_agent1, 'coins')
    own_kills1 = get_stat(own_agent1, 'kills')
    own_original_reward1 = own_coins1 * 1 + own_kills1 * 3

    own_coins2 = get_stat(own_agent2, 'coins')
    own_kills2 = get_stat(own_agent2, 'kills')
    own_original_reward2 = own_coins2 * 1 + own_kills2 * 3

    comparison = {
        'Metric': f"{own_agent_name1} ({filename1}) vs {own_agent_name2} ({filename2})",
        'Original Reward': f"{own_original_reward1} vs {own_original_reward2} ({own_original_reward1 - own_original_reward2:+})",
        'Score': f"{get_stat(own_agent1, 'score')} vs {get_stat(own_agent2, 'score')} ({get_stat(own_agent1, 'score') - get_stat(own_agent2, 'score'):+})",
        'Coins': f"{own_coins1} vs {own_coins2} ({own_coins1 - own_coins2:+})",
        'Kills': f"{own_kills1} vs {own_kills2} ({own_kills1 - own_kills2:+})",
        'Suicides': f"{get_stat(own_agent1, 'suicides')} vs {get_stat(own_agent2, 'suicides')} ({get_stat(own_agent1, 'suicides') - get_stat(own_agent2, 'suicides'):+})",
        'Invalid Actions': f"{get_stat(own_agent1, 'invalid')} vs {get_stat(own_agent2, 'invalid')} ({get_stat(own_agent1, 'invalid') - get_stat(own_agent2, 'invalid'):+})",
    }
    return comparison

def main():
    """
    Main function to parse command-line arguments and perform agent comparisons.
    """
    parser = argparse.ArgumentParser(description='Compare two JSON files with game statistics.')
    parser.add_argument('json1', help='First JSON file')
    parser.add_argument('json2', help='Second JSON file')
    args = parser.parse_args()

    data1 = load_json(args.json1)
    data2 = load_json(args.json2)

    # Compare own agent with others in the first file
    comparisons1 = compare_agents(data1, args.json1)
    # Compare own agent with others in the second file
    comparisons2 = compare_agents(data2, args.json2)
    # Compare the own agents between both files
    own_agents_comparison = compare_own_agents(data1, data2, args.json1, args.json2)

    print("\nComparison of own agent with other agents in file 1:")
    print(tabulate(comparisons1, headers="keys", tablefmt="grid"))

    print("\nComparison of own agent with other agents in file 2:")
    print(tabulate(comparisons2, headers="keys", tablefmt="grid"))
    
    print("\nComparison of own agent between both files:")
    print(tabulate([own_agents_comparison], headers="keys", tablefmt="grid"))

if __name__ == "__main__":
    main()
