#*****************************************************
#AUTHOR UTPAL SARKAR                                 *
#I am using Reinforcement Learning AI/ML technique   *
#basic code lot of enhancement needed                *
#USE CODE with own risk                              *
#This is for educational purpose                     *
#Leveraged Trading with  Volatility                  *
#                                                    *
#*****************************************************

import numpy as np
import pandas as pd
import argparse
import os


class LeveragedTradingAgent:
    def __init__(self, total_states, n_actions, leverage_factor=3, decay_rate=0.001, learning_rate=0.7, gamma=0.95):
        self.total_states = total_states
        self.n_actions = n_actions
        self.leverage_factor = leverage_factor
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = 1.0  # Initial exploration rate
        # Ensure Q-table is initialized with 'total_states' rows
        self.q_table = np.zeros((total_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:  # Exploration
            action = np.random.choice(self.n_actions)
        else:  # Exploitation
            action = np.argmax(self.q_table[state, :])
        return action

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state, :])
        self.q_table[state, action] += self.learning_rate * (target - predict)
        # Gradually decrease exploration rate
        self.epsilon = max(self.epsilon - self.decay_rate, 0.01)

#def get_state(data, t, n_states, n_vol_states):
#    # Normalize and discretize the price and volatility into their respective bins
#    price_state = int((data['Close'][t] - data['Close'].min()) / (data['Close'].max() - data['Close'].min()) * (n_states - 1))
#    vol_state = int((data['Volatility'][t] - data['Volatility'].min()) / (data['Volatility'].max() - data['Volatility'].min()) * (n_vol_states - 1))
#    # Calculate combined state index
#    combined_state = price_state * n_vol_states + vol_state
#    return combined_state

def get_state(data, t, n_states, n_vol_states, epsilon=1e-8):
    # Normalize and discretize the price into its bin
    price_range = data['Close'].max() - data['Close'].min()
    price_state = int((data['Close'][t] - data['Close'].min()) / (price_range + epsilon) * (n_states - 1))
    
    # Normalize and discretize the volatility into its bin
    vol_range = data['Volatility'].max() - data['Volatility'].min()
    vol_state = int((data['Volatility'][t] - data['Volatility'].min()) / (vol_range + epsilon) * (n_vol_states - 1))
    
    # Calculate combined state index
    combined_state = price_state * n_vol_states + vol_state
    return combined_state

def get_current_state(current_data, n_states, n_vol_states, epsilon=1e-8):
    # Assume 'current_data' is a dictionary or a similar structure holding the current observations
    # for price and volatility (or whatever features you're using)

    price_state = int((current_data['price'] - price_min) / (price_max - price_min + epsilon) * (n_states - 1))
    vol_state = int((current_data['volatility'] - vol_min) / (vol_max - vol_min + epsilon) * (n_vol_states - 1))

    combined_state = price_state * n_vol_states + vol_state
    return combined_state




def main(args):
  if args.csv_path is not None:
    df = pd.read_csv(args.csv_path)
    if args.csv_path is not None:
        # Calculate rolling volatility
        window_size = 11
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=window_size).std().fillna(0)
    
        g_volatility = df['Volatility'].max();
        print(f"Volatility =  {g_volatility}")


        n_states = 20
        n_vol_states = 20
        n_actions = 3
        total_states = n_states * n_vol_states

        agent = LeveragedTradingAgent(total_states, n_actions)

        q_table_path = 'learning_data/q_table.npy'
        if not os.path.exists('learning_data'):
            os.makedirs('learning_data')

        if os.path.exists(q_table_path):
            agent.q_table = np.load(q_table_path)

        n_episodes = 100
        risk_free_rate = 0.01 / 252  # Annualized risk-free rate converted to daily

        for episode in range(n_episodes):
            state = get_state(df, 0, n_states, n_vol_states)
            total_reward = 0

            for t in range(1, len(df)):
                action = agent.choose_action(state)
                next_state = get_state(df, t, n_states, n_vol_states)

                daily_return = df['Daily_Return'][t]
                reward = daily_return * agent.leverage_factor if action == 1 else -daily_return * agent.leverage_factor
                expected_return = daily_return - risk_free_rate
                reward = expected_return / df['Volatility'][t] if df['Volatility'][t] > 0 else expected_return

                total_reward += reward

                agent.learn(state, action, reward, next_state)
                state = next_state

            print(f"Episode {episode+1}: Total Reward: {total_reward}")



        np.save(q_table_path, agent.q_table)
        print("Training completed. Learned Q-values:")
        print(agent.q_table)



    
    if args.price is not None:
    	global price_min, price_max, vol_min, vol_max , vol_avg
    	price_min, price_max = df['Close'].min(), df['Close'].max()
    	vol_min, vol_max = df['Volatility'].min(), df['Volatility'].max()
    	vol_avg = (vol_max + vol_min)/2

    	print(f"The current volatility is: min: {vol_min} , max: {vol_max}  avg: {vol_avg}")
    	current_price = args.price
        #current_data = {'price': 50.22, 'volatility': 0.067281}  # Hypothetical current data
    	current_data = {'price': current_price, 'volatility': vol_max}  # Hypothetical current data

    	current_state = get_current_state(current_data, n_states, n_vol_states)
    	print(f"The current state based on the latest observations is: {current_state}")

    	# Determine the optimal action for the current state
    	q_values_current_state = agent.q_table[current_state]
    	optimal_action_index = np.argmax(q_values_current_state)
    	actions = ['Hold', 'Buy', 'Sell']  # Ensure this matches your agent's action definitions
    	optimal_action = actions[optimal_action_index]
    	print(f"Optimal action for the current state ({current_state}): {optimal_action}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Leveraged Trading with  Volatility')
    parser.add_argument('--csv_path', type=str, help='Path to the CSV file containing your data')
    parser.add_argument('--price', type=float, help='Specify the price')
    parser.add_argument('--train', type=str, help='Train option')

    args = parser.parse_args()
    main(args)

