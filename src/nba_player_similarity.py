import logging
from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data
def get_player_stats(season='2022-23'):
    """
    Fetch stats for all players from the specified season using a single API call.
    """
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base',
            plus_minus='N',
            pace_adjust='N',
            rank='N',
            season_type_all_star='Regular Season'
        )
        df = stats.get_data_frames()[0]
        logging.info(f"Successfully fetched data for {len(df)} players from {season} season")
        return df
    except Exception as e:
        logging.error(f"Error fetching player stats: {str(e)}")
        return None

def prepare_data(df):
    """
    Prepare the data for clustering by selecting relevant features.
    """
    features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    return df[['PLAYER_ID', 'PLAYER_NAME'] + features].dropna()

def cluster_players(df, n_clusters=5):
    """
    Perform K-means clustering on the player data.
    """
    features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    return df

def find_similar_players(df, player_name, top_n=5):
    """
    Find the top N similar players to the given player based on cluster and Euclidean distance.
    """
    player = df[df['PLAYER_NAME'] == player_name].iloc[0]
    cluster = player['Cluster']
    cluster_players = df[df['Cluster'] == cluster]
    
    features = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    player_stats = player[features].values
    
    distances = []
    for _, row in cluster_players.iterrows():
        dist = np.linalg.norm(player_stats - row[features].values)
        distances.append((row['PLAYER_NAME'], dist))
    
    distances.sort(key=lambda x: x[1])
    return [name for name, _ in distances[1:top_n+1]]  # Exclude the player himself

def main():
    st.title("NBA Player Similarity Finder")
    st.write("This app finds similar NBA players based on their stats.")

    # Fetch player stats
    with st.spinner("Fetching player data..."):
        df = get_player_stats()
    
    if df is None:
        st.error("Failed to fetch player data. Please try again later.")
        return

    # Prepare data
    df_prepared = prepare_data(df)
    st.success(f"Data prepared for {len(df_prepared)} players")

    # Perform clustering
    with st.spinner("Performing clustering..."):
        df_clustered = cluster_players(df_prepared)
    st.success("Clustering completed")

    # Player selection
    player_list = sorted(df_clustered['PLAYER_NAME'].tolist())
    selected_player = st.selectbox("Select a player:", player_list)
    
    if st.button("Find Similar Players"):
        similar_players = find_similar_players(df_clustered, selected_player)
        st.subheader(f"Players similar to {selected_player}:")
        for i, player in enumerate(similar_players, 1):
            st.write(f"{i}. {player}")

        # Display player stats
        st.subheader(f"Stats for {selected_player}:")
        player_stats = df_clustered[df_clustered['PLAYER_NAME'] == selected_player].iloc[0]
        group_stats = df_clustered[df_clustered['PLAYER_NAME'] == selected_player]
        stats_to_display = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
        for stat in stats_to_display:
            st.write(f"{stat}: {player_stats[stat]:.2f}")
        
        st.subheader(f"Stats for group including {selected_player}:")
        for stat in stats_to_display:
            print(group_stats)
            print(stat)
            st.write(f"{stat}: {group_stats.agg({{stat}:'mean'})}")
            
        #streamlit run nba_player_similarity.py

if __name__ == '__main__':
    main()