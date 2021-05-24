import pandas as pd
import streamlit as st
import sys
sys.path.append('../')

from utils import ui

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 2000px;
        padding-top: 0rem;
        padding-right: 15rem;
        padding-left: 15rem;
        padding-bottom: 0rem;
    }}
</style>
""",
        unsafe_allow_html=True,
    )


# load model prediction
result = pd.read_csv('../data/processed/last_result.csv')
result = result.loc[result.present_time == result.present_time.max(),
                   ].reset_index(drop=True)


result['pred_scenario'] = result.apply(lambda row : ui.get_str_scenario(row['pred_speed'],
                                                                        row['cos_wind_dir'],
                                                                        row['sin_wind_dir']),
                                       axis=1 )

result['pred_direction'] = result['pred_wind_dir'].map(ui.get_directions_label)

# format 
result['pred_speed'] = result['pred_speed'].map(lambda x : str(round(x,1)))
result['pred_wind_dir'] = result['pred_wind_dir'].map(lambda x : str(round(x)))
result['numtech_speed'] = result['numtech_speed'].map(lambda x : str(ui.round_numtech(x,1)))
result['numtech_wind_dir'] = result['numtech_wind_dir'].map(lambda x : str(ui.round_numtech(x,0)))

# header 
st.header("Prédiction du " + str(result['present_time'][0]))
# display
result = result[['datetime','pred_direction','pred_scenario',
                            'pred_speed','numtech_speed',
                            'pred_wind_dir','numtech_wind_dir']].set_index('datetime')

st.dataframe(result.style.applymap(ui.color_direction, subset=['pred_direction']), height=2000)