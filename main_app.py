import os
import time
import streamlit as st

from utils_info import *
from utils import *

def main():
    st.set_page_config(page_title='Music Generation', page_icon= "🖖")
    st.sidebar.title("Menu Options:")
    app_mode = st.sidebar.radio("Go to", ["Introduction", "Data Analysis",
                                          "Algorithm", "Model Description", "Predictions", "Performance Parameters"])

    if app_mode == "Data Analysis":
        st.title("Data Analysis of MIDI Files")
        select_action = st.sidebar.selectbox("Select function", ["Choose", "Data Info", 
                                                                "Raw MIDI", "Tokenized MIDI", "Play MIDI"])
        data_functions(select_action)
        #print_outline()
    
    elif app_mode == "Model Description":
        phase = st.sidebar.selectbox("Choose Phase", ["Phase I", "Phase II"])
        if phase == "Phase I":
            st.title("Phase I")
            print_lit_review()
            print_phase1()
        else:
            st.title("Phase II")
            print_phase2()
    
    elif app_mode == "Predictions":
        st.title("Music Generation Predictions")
        st.info("This section will help you listen and visualize the predictions/inferences made by the \
                trained model.")
        which_play = st.sidebar.selectbox("Choose Music Model", ["Reddit Pop Model", "Maestro Midi Model", "Theme Midi"])
        if which_play == "Maestro Midi Model":
            play_pred("pred_lmd")
        elif which_play == "Theme Midi":
            play_predd("pred_midi")
        else:
            play_pred("pred_reddit")

        #predictions()

    elif app_mode == "Performance Parameters":
        st.title ("Performance Paramters of the Songs and the Generated Output")
        perfparam()
    
    elif app_mode == "Algorithm":
        #st.header("Algorithm for the project")
        print_algo()

    elif app_mode == "Introduction":
        print_intro()
        #print_gui_info()
        intro()

if __name__ == "__main__":
    main()