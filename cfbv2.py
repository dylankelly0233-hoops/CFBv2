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
    api_key = st.text_input("CFBD API Key", value='VV6c9PpeO05qTuiMSlZZw6ijTEA0+E79bLPXBhsKuOhxSKn8wiYfIOX4U/ZNAok6', type="password")

    st.divider()
    year = st.number_input("Year", value=2025)
    target_week = st.number_input("Target Week", value=2)
    is_postseason = st.checkbox("Postseason Mode", value=True)

    st.divider()
    st.subheader("🏟️ Intelligent HFA")
    st.info("HFA scales based on Stadium Capacity.")
    min_hfa = st.slider("Min HFA (Small Stadium)", 0.5, 3.0, 1.5, help="HFA for ~20k capacity")
    max_hfa = st.slider("Max HFA (Massive Stadium)", 2.0, 6.0, 4.0, help="HFA for ~100k capacity")

    st.divider()
    st.subheader("Model Tuning")
    prior_weight = st.slider("SP+ Prior Weight", 0.0, 10.0, 5.0)
    decay_alpha = st.slider("Time Decay (Alpha)", 0.0, 0.5, 0.15)
    
    st.divider()
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

# --- DYNAMIC HFA LOGIC ---
def calculate_dynamic_hfa(team, team_capacity_map, min_val, max_val):
    """
    Scales HFA linearly based on capacity.
    Min HFA at 20,000 capacity. Max HFA at 100,000 capacity.
    """
    cap = team_capacity_map.get(team, 45000) # Default to 45k if unknown
    
    # Clip capacity to reasonable bounds
    cap = max(20000, min(cap, 100000))
    
    # Linear Interpolation
    # pct = (cap - min_cap) / (max_cap - min_cap)
    pct = (cap - 20000) / (80000) 
    
    # HFA = Min + (Range * Pct)
    hfa = min_val + ((max_val - min_val) * pct)
    return round(hfa, 2)

# --- MAIN LOGIC ---
if st.button("🚀 Run Model"):
    with st.spinner("Fetching Data & Stadium Info..."):
        # 1. FETCH DATA
        teams_json = get_cfbd_data('/teams', {'year': year})
        fbs_teams = {t['school'] for t in teams_json if t.get('classification') == 'fbs'}
        
        # Build Capacity Map (Team -> Venue Capacity)
        team_capacity_map = {}
        for t in teams_json:
            school = t['school']
            loc = t.get('location', {})
            cap = loc.get('capacity')
            if cap:
                team_capacity_map[school] = int(cap)
            else:
                team_capacity_map[school] = 45000 # Average FBS Fallback

        games_reg = get_cfbd_data('/games', {'year': year, 'seasonType': 'regular'})
        games_post = get_cfbd_data('/games', {'year': year, 'seasonType': 'postseason'})

        # Map Neutral Sites
        game_info_map = {}
        for g in (games_reg + games_post):
            h, a = g.get('home_team'), g.get('away_team')
            game_info_map[f"{h}_{a}"] = g.get('neutral_site', False)

        lines_reg = get_cfbd_data('/lines', {'year': year, 'seasonType': 'regular'})
        for g in lines_reg: g['_season_type'] = 'regular'
        lines_post = get_cfbd_data('/lines', {'year': year, 'seasonType': 'postseason'})
        for g in lines_post: g['_season_type'] = 'postseason'
        lines_raw = lines_reg + lines_post

        sp_json = get_cfbd_data('/ratings/sp', {'year': year})
        sp_map = {item['team']: item['rating'] for item in sp_json}

        # 2. BUILD TRAINING SET
        matchups = []
        effective_week = 15 if is_postseason else target_week
        current_prior_weight = max(0.1, prior_weight * (0.75 ** effective_week))

        for g in lines_raw:
            week = g.get('week', 0)
            sType = g.get('_season_type')
            home, away = g.get('homeTeam'), g.get('awayTeam')

            # Filter FCS vs FCS
            if home not in fbs_teams and away not in fbs_teams: continue
            
            # Filter Future Games
            if is_postseason:
                if sType == 'postseason' and week >= target_week: continue
            else:
                if sType == 'postseason': continue
                if week >= target_week: continue

            lines = g.get('lines', [])
            if not lines: continue
            spread_val = None
            for p in lines:
                if p.get('spread') is not None:
                    spread_val = p.get('spread')
                    break
            if spread_val is None: continue

            is_neutral = game_info_map.get(f"{home}_{away}", False)
            
            # --- INTELLIGENT HFA ADJUSTMENT ---
            # To train a "Neutral" rating, we must remove the HFA from the result/spread first.
            if is_neutral:
                this_game_hfa = 0.0
            else:
                this_game_hfa = calculate_dynamic_hfa(home, team_capacity_map, min_hfa, max_hfa)

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
                'hfa_val': this_game_hfa, # Store what we used
                'weight': game_weight
            })

        # Inject Priors
        for team, rating in sp_map.items():
            if team in fbs_teams:
                matchups.append({
                    'home_team': team,
                    'away_team': 'LEAGUE_AVERAGE_DUMMY',
                    'spread': -1 * rating,
                    'hfa_val': 0.0, # Neutral prior
                    'weight': current_prior_weight
                })

        df = pd.DataFrame(matchups)
        if df.empty:
            st.error("No data found.")
            st.stop()

        # 3. RIDGE REGRESSION
        # We adjust the target variable to remove HFA impact
        # Implied Margin = (Home Score - Away Score)
        # We want: Neutral Margin = Implied Margin - Home_HFA
        # Market Spread of -7 means Home is favored by 7 (Margin +7)
        
        df['implied_margin'] = -1 * df['spread']
        df['neutral_margin'] = df['implied_margin'] - df['hfa_val']

        home_dummies = pd.get_dummies(df['home_team'], dtype=int)
        away_dummies = pd.get_dummies(df['away_team'], dtype=int)
        all_teams = sorted(list(set(home_dummies.columns) | set(away_dummies.columns)))
        if 'LEAGUE_AVERAGE_DUMMY' in all_teams: all_teams.remove('LEAGUE_AVERAGE_DUMMY')

        home_dummies = home_dummies.reindex(columns=all_teams, fill_value=0)
        away_dummies = away_dummies.reindex(columns=all_teams, fill_value=0)

        X = home_dummies.sub(away_dummies)
        # We NO LONGER solve for HFA intercept, because we manually removed it!
        # This makes the ratings "Pure Neutral"
        y = df['neutral_margin']

        w_array = df['weight'].values
        w_normalized = w_array * (len(w_array) / w_array.sum())

        clf = Ridge(alpha=0.0001, fit_intercept=False)
        clf.fit(X, y, sample_weight=w_normalized)

        coefs = pd.Series(clf.coef_, index=X.columns)
        team_ratings = coefs - coefs.mean()
        
        st.success(f"Model Trained using Intelligent HFA ({min_hfa} to {max_hfa} pts)")

        # 4. PREPARE PROJECTIONS TABLE (Pre-Calculation)
        proj_data = []
        target_slate = [
            g for g in lines_raw
            if g.get('_season_type') == ('postseason' if is_postseason else 'regular')
               and g.get('week') == target_week
        ]

        for g in target_slate:
            home, away = g.get('homeTeam'), g.get('awayTeam')
            if home not in fbs_teams and away not in fbs_teams: continue

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
            
            # Calculate Specific HFA for this matchup
            hfa = 0.0 if is_neutral else calculate_dynamic_hfa(home, team_capacity_map, min_hfa, max_hfa)

            proj_data.append({
                "Away Team": away,
                "Home Team": home,
                "Vegas Line": vegas_spread,
                "Home HFA": hfa,
                "Rating Away": round(a_r, 1),
                "Rating Home": round(h_r, 1),
                "Away QB Adj": 0.0, # Default User Input
                "Home QB Adj": 0.0  # Default User Input
            })
            
        proj_df_raw = pd.DataFrame(proj_data)

        # 5. RENDER INTERACTIVE TABLE (User adjusts QBs here)
        st.subheader("🛠️ Matchup Adjustments")
        st.caption("Instructions: Use 'Away QB Adj' and 'Home QB Adj' to penalize teams (e.g., enter -4.5 if starter is out).")
        
        edited_proj_df = st.data_editor(
            proj_df_raw,
            column_config={
                "Away QB Adj": st.column_config.NumberColumn("Away QB Adj", help="Penalty for Away Backup (e.g. -3.0)", format="%.1f", step=0.5),
                "Home QB Adj": st.column_config.NumberColumn("Home QB Adj", help="Penalty for Home Backup (e.g. -3.0)", format="%.1f", step=0.5),
                "Vegas Line": st.column_config.NumberColumn("Vegas Line", format="%.1f"),
                "Away Team": st.column_config.TextColumn("Away Team", disabled=True),
                "Home Team": st.column_config.TextColumn("Home Team", disabled=True),
                "Home HFA": st.column_config.NumberColumn("HFA", format="%.2f", disabled=True),
            },
            hide_index=True,
            use_container_width=True
        )

        # 6. CALCULATE FINAL RESULTS (Live Update)
        results = []
        for _, row in edited_proj_df.iterrows():
            h_r = row['Rating Home']
            a_r = row['Rating Away']
            hfa = row['Home HFA']
            v_line = row['Vegas Line']
            
            # User Adjustments
            a_adj = row['Away QB Adj']
            h_adj = row['Home QB Adj']
            
            # Final Model Calculation
            # Spread = (Home_Rating + Home_Adj) - (Away_Rating + Away_Adj) + HFA
            # We treat Ratings as "Points above Average" (Positive = Good)
            # So Home - Away gives Home Margin.
            
            raw_margin = (h_r + h_adj) - (a_r + a_adj) + hfa
            model_spread = -raw_margin # Convert to Betting Line (Home -7)
            
            # Edge (Model Margin vs Vegas Margin)
            model_margin_positive = raw_margin
            vegas_margin_positive = -v_line
            
            edge = model_margin_positive - vegas_margin_positive
            
            # Thresholds
            is_key = False
            if (model_spread < -3 and v_line > -3) or (model_spread > -3 and v_line < -3): is_key = True
            if (model_spread < -7 and v_line > -7) or (model_spread > -7 and v_line < -7): is_key = True
            req_edge = thresh_key if is_key else thresh_std
            
            signal = "PASS"
            if edge > req_edge: signal = f"BET {row['Home Team']}"
            elif edge < -req_edge: signal = f"BET {row['Away Team']}"
            
            results.append({
                "Matchup": f"{row['Away Team']} @ {row['Home Team']}",
                "Vegas": v_line,
                "Model": round(model_spread, 1),
                "Edge": round(edge, 1),
                "Signal": signal
            })
            
        final_df = pd.DataFrame(results)
        
        col1, col2 = st.columns([1, 1])
        with col1:
             st.subheader("📊 Team Ratings")
             ratings_df = pd.DataFrame({'Team': team_ratings.index, 'Rating': team_ratings.values})
             ratings_df = ratings_df.sort_values('Rating', ascending=False).reset_index(drop=True)
             ratings_df.index += 1
             st.dataframe(ratings_df, height=500, use_container_width=True)
             
        with col2:
            st.subheader("💰 Final Betting Board")
            def color_signal(val):
                color = 'white'
                if "BET" in str(val): color = '#d4edda'
                return f'background-color: {color}; color: black'
            
            st.dataframe(
                final_df.style.applymap(color_signal, subset=['Signal'])
                .format({"Vegas": "{:.1f}", "Model": "{:.1f}", "Edge": "{:.1f}"}),
                height=500,
                use_container_width=True
            )
            
            # Download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, sheet_name='Picks', index=False)
                ratings_df.to_excel(writer, sheet_name='Ratings', index=False)
                edited_proj_df.to_excel(writer, sheet_name='Adjustments_Used', index=False)
                
            st.download_button("📥 Download Excel", buffer.getvalue(), f"CFB_W{target_week}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
