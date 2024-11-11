import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, random, lax
import seaborn as sns
def min_to_hour(df_temp):
    df_temp['hourly'] = df_temp['created_at'].dt.floor('H')
    df_temp['daily'] = df_temp['created_at'].dt.floor('D')

    df_temp2 = df_temp[['hourly', 'daily', 'user_id', 'prize_value', 'Tcs', 'Tin', 'Tout', 'RH_in', 'RH_out']]

    grouped_hourly = df_temp2.groupby(['hourly', 'user_id']).agg({
        'prize_value': lambda x: x.mode()[0],
        'Tcs': lambda x: x.mode()[0],
        'Tin': 'mean',
        'Tout': 'mean',
        'RH_in': 'mean',
        'RH_out': 'mean'
    }).reset_index()

    grouped_daily = df_temp2.groupby(['daily', 'user_id']).agg({
        'Tout': 'mean',
        'RH_out': 'mean'
    }).reset_index().rename(columns={'Tout': 'Tout_daily', 'RH_out': 'RH_out_daily'})

    grouped_modes = pd.merge(
        grouped_hourly,
        grouped_daily[['daily', 'user_id', 'Tout_daily', 'RH_out_daily']],
        left_on=['user_id'],
        right_on=['user_id'],
        how='left'
    )

    grouped_modes = grouped_modes[(grouped_modes['hourly'].dt.floor('D') == grouped_modes['daily'])]
    return grouped_modes

@jit
def log_likelihood(omega, Fs, features_numeric):
    rewards = jnp.dot(features_numeric, omega)
    exp_rewards = jnp.exp(rewards)
    total = jnp.sum(exp_rewards)
    ll = jnp.dot(Fs, rewards) - jnp.log(total) * jnp.sum(Fs)
    return ll
def gradient_ascent(Fs, omega_initial, features_numeric, lr):
    omega = omega_initial
    path = []
    log_likelihood_history = []
    grad_history = []
    for i in range(10000):
        exp_rewards = jnp.exp(jnp.dot(features_numeric, omega))
        total = jnp.sum(exp_rewards)
        grad = jnp.dot(features_numeric, Fs - (exp_rewards / total))
        omega += lr * grad
        ll = log_likelihood(omega, Fs, features_numeric)

        path.append(omega)
        log_likelihood_history.append(ll)
        grad_history.append(grad)
        print(i)
    return omega, path, log_likelihood_history, grad_history


df = pd.read_feather('df_integrated.feather')
users_anonymized = [57, 58, 63, 64, 66, 67, 69, 70, 71, 72, 76, 81, 83, 87, 88, 91, 94, 95, 98, 99, 100, 102]
df = df[df['user_id'].isin(users_anonymized)]
df = df[(df['created_at']>='2024-07-01') & (df['created_at']<='2024-09-01')]
df = df.dropna(subset=['mode', 'Tin'])
df_processed = min_to_hour(df.copy())

cols_to_convert = ['Tcs', 'Tin', 'Tout', 'Tout_daily', 'RH_in', 'RH_out', 'RH_out_daily']
df_processed[cols_to_convert] = df_processed[cols_to_convert].astype(int)
df_processed['state_action'] = df_processed.apply(
    lambda x: f"{x['Tin']} - {x['Tout']} - {x['Tout_daily']} - {x['RH_in']} - {x['RH_out']} - {x['RH_out_daily']} - {x['prize_value']} - {x['Tcs']}", axis=1
)


df_final = pd.DataFrame()
df_log_likelihood_history = pd.DataFrame(columns=['history', 'user_id', 'iteration'])

for user_id in np.sort(df_processed['user_id'].unique()):
    df_user = df_processed[(df_processed['user_id']==user_id)].sort_values(by='hourly')

    Fs = jnp.array(df_user['state_action'].value_counts(normalize=True))
    states = df_user['state_action'].unique()
    state_indices = {state: i for i, state in enumerate(states)}
    states_numeric = jnp.array(list(state_indices.values()))

    features = {state: jnp.eye(len(states))[i] for i, state in enumerate(states)}
    features_numeric = jnp.array([features[state] for state in states])

    omega_initial = random.normal(random.PRNGKey(0), (features_numeric.shape[1],))
    optimal_omega, path, log_likelihood_history, grad_history = gradient_ascent(Fs, omega_initial, features_numeric, lr=0.2)

    omega_df = pd.DataFrame({
        'State_Action': states,
        'Optimal_Omega': optimal_omega
    })
    split_columns = omega_df['State_Action'].str.split(' - ', expand=True).astype(float)
    split_columns.columns = ['Tin', 'Tout', 'Tout_daily', 'RH_in', 'RH_out', 'RH_out_daily', 'prize_value', 'Tcs']

    df_final_temp = pd.concat([split_columns, omega_df], axis=1)
    df_final_temp['user_id'] = user_id
    df_final = pd.concat([df_final, df_final_temp], ignore_index=True)

    df_log_likelihood_history_temp = pd.DataFrame([float(value.item()) for value in log_likelihood_history], columns=['history'])
    df_log_likelihood_history_temp = df_log_likelihood_history_temp.reset_index().rename(columns={'index': 'iteration'})
    df_log_likelihood_history_temp['user_id'] = user_id
    df_log_likelihood_history = pd.concat([df_log_likelihood_history, df_log_likelihood_history_temp], ignore_index=True)

df_final = df_final.reset_index(drop=True)
df_log_likelihood_history = df_log_likelihood_history.reset_index(drop=True)
df_final.to_feather('df_final.feather')
df_log_likelihood_history.to_feather('df_log_likelihood_history.feather')
