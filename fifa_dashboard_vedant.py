# -*- coding: utf-8 -*-

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from math import pi

# --- Streamlit Page Setup ---
st.set_page_config(page_title="⚽ FIFA Scouting Dashboard", layout="wide")

# Custom CSS to shrink sidebar width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            width: 220px;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 220px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS to reduce Budget Remaining font size
st.markdown(
    """
    <style>
        div[data-testid="metric-container"] > div > div > span {
            font-size: 12px;  /* Change to whatever size you prefer */
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Load Data ---
df = pd.read_csv("fifa_players.csv")

# Drop duplicates
df.drop_duplicates(inplace=True)

# Drop unnecessary columns
national_cols = ['national_team', 'national_rating', 'national_team_position', 'national_jersey_number']
df.drop(columns=national_cols, inplace=True)

# Drop rows missing key financial info
df.dropna(subset=['value_euro', 'wage_euro'], inplace=True)

# Retain important columns
columns_to_keep = [
    'name', 'age', 'height_cm', 'weight_kgs', 'positions', 'nationality', 'body_type', 'overall_rating', 'potential', 'value_euro',
    'crossing', 'finishing', 'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve', 'freekick_accuracy', 'long_passing',
    'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina', 'strength',
    'long_shots', 'aggression', 'interceptions', 'positioning', 'vision', 'penalties', 'composure', 'marking', 'standing_tackle', 'sliding_tackle'
]
df = df[columns_to_keep]

# Assign Position Groups
def assign_position_group(pos):
    if 'GK' in pos:
        return 'Goalkeeper'
    elif any(p in pos for p in ['CB', 'LB', 'RB', 'LWB', 'RWB']):
        return 'Defender'
    elif any(p in pos for p in ['CM', 'CDM', 'CAM', 'RM', 'LM', 'LW', 'RW']):
        return 'Midfielder'
    elif any(p in pos for p in ['ST', 'CF', 'LF', 'RF']):
        return 'Forward'
    else:
        return 'Other'
    
df['position_group'] = df['positions'].apply(assign_position_group)

# Fill missing numeric values
feature_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

# Clustering
features = df.select_dtypes(include=[np.number]).drop(columns=['age', 'height_cm', 'weight_kgs', 'value_euro'])
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

# Select top players
top_players = {}
for group in ['Forward', 'Midfielder', 'Defender', 'Goalkeeper']:
    group_df = df[df['position_group'] == group]
    best_cluster = group_df.groupby('cluster')['overall_rating'].mean().idxmax()
    top_players[group] = group_df[group_df['cluster'] == best_cluster].sort_values(by='overall_rating', ascending=False)

# --- Streamlit App ---

# Centered Title
st.markdown("<h1 style='text-align: center; color: gold;'>⚽ FIFA Scouting Dashboard ⚽</h1>", unsafe_allow_html=True)

# Session State Initialization
if "budget" not in st.session_state:
    st.session_state.budget = 1_000_000_000
if "purchased_players" not in st.session_state:
    st.session_state.purchased_players = []
if "purchased_by_role" not in st.session_state:
    st.session_state.purchased_by_role = {'Forward': [], 'Midfielder': [], 'Defender': [], 'Goalkeeper': []}
if "team_overall" not in st.session_state:
    st.session_state.team_overall = []

# Sidebar

with st.sidebar:
    st.header("⚙️ Squad Builder")
    selected_role = st.selectbox("Select Role", list(top_players.keys()))
    role_df = top_players[selected_role].head(30)

    player_display = [f"{row['name']} ({int(row['overall_rating'])})" for idx, row in role_df.iterrows()]
    selected_display_name = st.selectbox("Select Player", player_display)
    selected_player_name = selected_display_name.split(' (')[0]

    selected_player_row = role_df[role_df['name'] == selected_player_name].iloc[0]

    if st.button("✅ Purchase Player", key="purchase_button"):
        if selected_player_name in st.session_state.purchased_players:
            st.warning(f"{selected_player_name} is already purchased.")
        elif len(st.session_state.purchased_players) >= 11:
            st.error("You’ve already purchased 11 players.")
        elif st.session_state.budget >= selected_player_row['value_euro']:
            st.session_state.budget -= selected_player_row['value_euro']
            st.session_state.purchased_players.append(selected_player_name)
            st.session_state.purchased_by_role[selected_role].append(selected_player_name)
            st.session_state.team_overall.append(selected_player_row['overall_rating'])
            st.success(f"🎉 {selected_player_name} added to squad!")
        else:
            st.error("Not enough budget.")

    st.subheader("🧾 Purchased Players")
    for role, players in st.session_state.purchased_by_role.items():
        if players:
            st.markdown(f"**{role}:**")
            for p in players:
                st.markdown(f"- {p}")

    if st.button("🔄 Reset Squad", key="reset_button"):
        st.session_state.budget = 1_000_000_000
        st.session_state.purchased_players = []
        st.session_state.purchased_by_role = {'Forward': [], 'Midfielder': [], 'Defender': [], 'Goalkeeper': []}
        st.session_state.team_overall = []
        st.success("Squad Reset!")

# --- Main Layout: Left = Budget + Overall Rating + Player Metrics | Right = Radar Chart ---

col1, col2 = st.columns([2, 3])  # Left narrow, right wide

# LEFT SIDE - col1
with col1:

    # Player Metric Cards
    st.subheader(f"👤 {selected_player_name}")

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric(label="Overall", value=int(selected_player_row['overall_rating']))
    with mcol2:
        st.metric(label="Age", value=int(selected_player_row['age']))
    with mcol3:
        st.metric(label="Height (cm)", value=int(selected_player_row['height_cm']))

    mcol4, mcol5 = st.columns(2)
    with mcol4:
        st.metric(label="Weight (kg)", value=int(selected_player_row['weight_kgs']))
    with mcol5:
        st.metric(label="Value (€)", value=f"€{int(selected_player_row['value_euro']):,}")

    st.markdown("---")  # Divider

    # Budget + Overall Rating

    st.subheader("💰 Budget Remaining")
    st.metric(label="", value=f"€{st.session_state.budget:,}")
    
    st.subheader("⭐ Squad Overall Rating")
    if st.session_state.team_overall:
        avg_overall = np.mean(st.session_state.team_overall)
        st.metric(label="", value=f"{avg_overall:.1f}")
    else:
        st.metric(label="", value="N/A")


with col2:


    stats = ['crossing', 'finishing', 'volleys', 'dribbling', 'curve',
             'ball_control', 'agility', 'reactions', 'balance', 'shot_power',
             'jumping', 'stamina', 'strength', 'long_shots', 'aggression',
             'interceptions', 'positioning', 'vision', 'penalties', 'composure',
             'marking', 'standing_tackle', 'sliding_tackle']

    values = [selected_player_row[stat] for stat in stats]
    num_vars = len(stats)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))  # ⬅️ smaller figure
    fig.patch.set_facecolor('#1D1D1D')
    ax.set_facecolor('#1D1D1D')

    # Fill and border
    ax.fill(angles, values, color='#FFD700', alpha=0.4)
    ax.plot(angles, values, color='#FF8C00', linewidth=1.5)  # ⬅️ thinner lines

    # Radar Settings
    ax.set_ylim(0, 100)
    ax.spines['polar'].set_visible(False)
    ax.set_yticklabels([])

    # Label Settings
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(stats, rotation=45, ha='center', fontsize=6, color='white')  # ⬅️ smaller font
    ax.tick_params(axis='x', pad=1)  # ⬅️ closer to center

    st.pyplot(fig)

