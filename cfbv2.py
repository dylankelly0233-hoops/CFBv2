import streamlit as st
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
import io

# --- PAGE CONFIG ---
st.set_page_config(page_title="CFB Power Ratings", layout="wide")
st.title("🏈 College Football Market Ratings & Betting Tool")

# --- SIDEBAR: CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # API Key Input (Defaults to the one you provided for convenience)
    api_key = st.text_input("CFBD API Key", value='VV6c9PpeO05qTuiMSlZZw6ijTEA0+E79bLPXBhsKuOhxSKn8wiYfIOX4U/ZNAok6',
                            type="password")

    st.divider()
    year = st.number_input("Year", value=2025)
    target_week = st.number_input("Target Week", value=2)
    is_postseason = st.checkbox("Postseason Mode", value=True)

    st.divider()
    st.subheader("Model Tuning")
    prior_weight = st.slider("SP+ Prior Weight", 0.0, 10.0, 5.0, help="How much we trust SP+ vs Actual Games")
    decay_alpha = st.slider("Time Decay (Alpha)", 0.0, 0.5, 0.15, help="Higher = Recency matters more")

    st.divider()
    st.subheader("Betting Thresholds")
    thresh_std = st.number_input("Standard Edge Req.", value=2.5, step=0.5)
    thresh_key = st.number_input("Key Number Edge Req.", value=1.5, step=0.5)

# --- API HELPERS ---
HEADERS = {'Authorization': f'Bearer {api_key}', 'accept': 'application/json'}
BASE_URL = 'https://api.collegefootballdata.com'


@st.cache_data(ttl=3600)
def get_cfbd_data(endpoint, params=None):
    try:
        response = requests.get(f"{BASE_URL}{endpoint}", headers=HEADERS, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching {endpoint}: {e}")
        return []


# --- MAIN LOGIC ---
if st.button("🚀 Run Model"):
    with st.spinner("Fetching Data from CFBD..."):
        # 1. FETCH DATA
        # A. Get Team Info (To filter FBS)
        teams_json = get_cfbd_data('/teams', {'year': year})
        # Create a set of ONLY FBS schools
        fbs_teams = {t['school'] for t in teams_json if t.get('classification') == 'fbs'}
        
        # B. Get Games
        games_reg = get_cfbd_data('/games', {'year': year, 'seasonType': 'regular'})
        games_post = get_cfbd_data('/games', {'year': year, 'seasonType': 'postseason'})

        # Map Neutral Sites
        game_info_map = {}
        for g in (games_reg + games_post):
            h, a = g.get('home_team'), g.get('away_team')
            game_info_map[f"{h}_{a}"] = g.get('neutral_site', False)

        # C. Fetch Lines
        lines_reg = get_cfbd_data('/lines', {'year': year, 'seasonType': 'regular'})
        for g in lines_reg: g['_season_type'] = 'regular'

        lines_post = get_cfbd_data('/lines', {'year': year, 'seasonType': 'postseason'})
        for g in lines_post: g['_season_type'] = 'postseason'

        lines_raw = lines_reg + lines_post

        # D. Fetch SP+ Priors
        sp_json = get_cfbd_data('/ratings/sp', {'year': year})
        sp_map = {item['team']: item['rating'] for item in sp_json}

        # 2. BUILD TRAINING SET
        matchups = []
        effective_week = 15 if is_postseason else target_week

        # Decay Calculation
        current_prior_weight = max(0.1, prior_weight * (0.75 ** effective_week))

        # A. Process Games
        count_games = 0
        for g in lines_raw:
            week = g.get('week', 0)
            sType = g.get('_season_type')
            
            home, away = g.get('homeTeam'), g.get('awayTeam')

            # --- FILTER: REMOVE FCS vs FCS ---
            # If NEITHER team is in our fbs_teams list, skip the game.
            if home not in fbs_teams and away not in fbs_teams:
                continue

            # Standard Logic
            if is_postseason:
                if sType == 'postseason' and week >= target_week: continue
            else:
                if sType == 'postseason': continue
                if week >= target_week: continue

            lines = g.get('lines', [])
            if not lines: continue

            # Find a valid provider
            spread_val = None
            for provider in lines:
                if provider.get('spread') is not None:
                    spread_val = provider.get('spread')
                    break

            if spread_val is None: continue

            is_neutral = game_info_map.get(f"{home}_{away}", False)

            # Weight Decay
            if sType == 'regular':
                weeks_ago = (15 - week) + target_week if is_postseason else (target_week - week)
            else:
                weeks_ago = target_week - week

            game_weight = np.exp(-decay_alpha * max(0, weeks_ago))

            matchups.append({
                'home_team': home,
                'away_team': away,
                'spread': float(spread_val),
                'is_neutral': is_neutral,
                'weight': game_weight
            })
            count_games += 1

        # B. Inject Priors (SP+)
        if count_games == 0:
            current_prior_weight = 10.0

        for team, rating in sp_map.items():
            # Only add priors for FBS teams to keep the matrix clean
            if team in fbs_teams:
                matchups.append({
                    'home_team': team,
                    'away_team': 'LEAGUE_AVERAGE_DUMMY',
                    'spread': -1 * rating,
                    'is_neutral': True,
                    'weight': current_prior_weight
                })

        df = pd.DataFrame(matchups)

        if df.empty:
            st.error("No data found for the selected parameters.")
            st.stop()

        # 3. RIDGE REGRESSION
        df['implied_margin'] = -1 * df['spread']
        home_dummies = pd.get_dummies(df['home_team'], dtype=int)
        away_dummies = pd.get_dummies(df['away_team'], dtype=int)

        all_teams = sorted(list(set(home_dummies.columns) | set(away_dummies.columns)))
        if 'LEAGUE_AVERAGE_DUMMY' in all_teams: all_teams.remove('LEAGUE_AVERAGE_DUMMY')

        home_dummies = home_dummies.reindex(columns=all_teams, fill_value=0)
        away_dummies = away_dummies.reindex(columns=all_teams, fill_value=0)

        X = home_dummies.sub(away_dummies)
        X['HFA_Constant'] = df['is_neutral'].apply(lambda x: 0 if x else 1)
        y = df['implied_margin']

        w_array = df['weight'].values
        w_normalized = w_array * (len(w_array) / w_array.sum())

        clf = Ridge(alpha=0.0001, fit_intercept=False)
        clf.fit(X, y, sample_weight=w_normalized)

        coefs = pd.Series(clf.coef_, index=X.columns)
        implied_hfa = coefs['HFA_Constant']
        team_ratings = coefs.drop('HFA_Constant')
        team_ratings = team_ratings - team_ratings.mean()

        # 4. GENERATE PROJECTIONS
        st.success(f"Model Trained! Market HFA detected: {implied_hfa:.2f} points")

        projections = []
        target_slate = [
            g for g in lines_raw
            if g.get('_season_type') == ('postseason' if is_postseason else 'regular')
               and g.get('week') == target_week
        ]

        for g in target_slate:
            home, away = g.get('homeTeam'), g.get('awayTeam')
            
            # --- FILTER TARGET SLATE TOO ---
            if home not in fbs_teams and away not in fbs_teams:
                continue

            lines = g.get('lines', [])
            if not lines: continue
            spread_val = None
            for p in lines:
                if p.get('spread') is not None:
                    spread_val = p.get('spread')
                    break
            if spread_val is None: continue
            vegas_spread = float(spread_val)

            h_r = team_ratings.get(home, 0.0)
            a_r = team_ratings.get(away, 0.0)

            is_neutral = game_info_map.get(f"{home}_{away}", True)
            hfa = 0.0 if is_neutral else implied_hfa

            raw_margin = h_r - a_r + hfa
            model_spread = -raw_margin

            model_margin = raw_margin
            vegas_margin = -vegas_spread

            edge = model_margin - vegas_margin
            
            is_key_cross = False
            m_spread = model_spread
            v_spread = vegas_spread

            if (m_spread < -3 and v_spread > -3) or (m_spread > -3 and v_spread < -3): is_key_cross = True
            if (m_spread < -7 and v_spread > -7) or (m_spread > -7 and v_spread < -7): is_key_cross = True
            if (m_spread < 3 and v_spread > 3) or (m_spread > 3 and v_spread < 3): is_key_cross = True

            req_edge = thresh_key if is_key_cross else thresh_std

            signal = "PASS"
            if edge > req_edge:
                signal = f"BET {home}"
            elif edge < -req_edge:
                signal = f"BET {away}"

            projections.append({
                "Matchup": f"{away} @ {home}",
                "Vegas Line": vegas_spread,
                "Model Line": round(model_spread, 1),
                "Edge": round(edge, 1),
                "Req Edge": req_edge,
                "Signal": signal,
                "Rating Away": round(a_r, 1),
                "Rating Home": round(h_r, 1)
            })

        # 5. DISPLAY
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("📊 Team Ratings")
            ratings_df = pd.DataFrame({'Team': team_ratings.index, 'Rating': team_ratings.values})
            ratings_df = ratings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
            ratings_df.index += 1
            st.dataframe(ratings_df, height=600, use_container_width=True)

        with col2:
            st.subheader("💰 Betting Board")
            proj_df = pd.DataFrame(projections)
            if not proj_df.empty:
                def color_signal(val):
                    color = 'white'
                    if "BET" in str(val):
                        color = '#d4edda'
                    return f'background-color: {color}; color: black'

                st.dataframe(
                    proj_df.style.applymap(color_signal, subset=['Signal'])
                    .format({"Vegas Line": "{:.1f}", "Model Line": "{:.1f}", "Edge": "{:.1f}", "Req Edge": "{:.1f}"}),
                    height=600,
                    use_container_width=True
                )

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    proj_df.to_excel(writer, sheet_name='Projections', index=False)
                    ratings_df.to_excel(writer, sheet_name='Ratings', index=False)

                st.download_button(
                    label="📥 Download Excel Report",
                    data=buffer.getvalue(),
                    file_name=f"CFB_Projections_Week_{target_week}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("No games found for this week yet.")

