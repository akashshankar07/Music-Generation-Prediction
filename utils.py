import os
import time
import threading
from multiprocessing import Process, Manager

from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from music21 import converter, instrument, note, chord, stream

import miditok,  miditoolkit

#from musicautobot.music_transformer.transform import *

#from musicautobot.vocab import MusicVocab
from utils_info import *
from pathlib import Path

def play_updated(file):
    audio_file = open(file, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

"""def append_to_list(L, file_name, vocab):

    for f in file_name:
        try:
            item = MusicItem.from_file(f, vocab)
            L.append(f)
        except:
            pass
"""

def get_files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


def divide_chunks(l, n): 
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n]

def data_functions(select_action):

    if select_action == "Choose":
        print_choose()

    elif select_action == "Data Info":
        st.info("This section will give insight about data gathering and analytics")
        # info_checkbox = st.checkbox("Display information about the MIDI data")
        data_info_visual()

    else:
        midi_file, org_file, mp3_ex, vocab = data_analysis_init()

        if select_action == "Raw MIDI":
            raw_midi(midi_file)

        elif select_action == "Tokenized MIDI":
            st.info("This section will explain how the tokenization works")

            tokenize_midi(midi_file)

        elif select_action == "Play MIDI":
            music_midi(mp3_ex, org_file)

def data_info_visual():

    #Nottingham Music Database,ABC-Notation Database
    #Maestro, Schubbert

    data_set_all = ["Nottingham Dataset", "Maestro Dataset", "Others"]
    sizes = [1200, 4000, 1000]
    colors = ['gold', 'lightblue', 'yellowgreen']
    explode = (0.0, 0.0, 0.0)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    make_chart(type_data= "Total Data", labels=data_set_all, 
                sizes=sizes, colors=colors, explode=explode)
    
    st.write("MAESTRO (MIDI and Audio Edited for Synchronous Tracks and Organization) is a dataset composed of about 200 hours of virtuosic piano performances captured with fine alignment (~3 ms) between note labels and audio waveforms.")
    st.write("Nottingham Dataset is a collection of 1200 British and American folk tunes, (hornpipe, jigs, and etc.) that was created by Eric Foxley and posted on Eric Foxley's Music Database. The database was converted to abc music notation format and was posted on abc.sourceforge.net.")

    data_sets = ["Schubert", "Piano-midi.de", "Ambrose piano", "MPI Dataset", "Reddit Dataset"]
    numbers = [90, 270, 50, 60, 750]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'green']  
    make_chart(type_data="Division of 'Others' Dataset",
                labels= data_sets, sizes=numbers,
                colors = colors,
                explode = (0.1, 0.1, 0.1, 0.1, 0.1))
    st.write("##")
    st.write("These include the other datasets taken into consideration")


def make_chart(type_data = "Data", 
			   labels = ['Python', 'C++', 'Ruby', 'Java', 'Haskell'],
			   sizes=[215, 130, 245, 210, 100], 
			   colors=['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'green'],
			   explode = (0.1, 0.1, 0.1, 0.1, 0.1)):

	st.subheader(type_data)

	# Plot
	plt.pie(sizes, explode=explode, labels=labels, colors=colors,
	autopct='%1.1f%%', shadow=True, startangle=140)
    
	plt.axis('equal')
    
	st.pyplot()


def data_analysis_init():

    #vocab = MusicVocab.create()
    mp3_path = Path("./streamlit_data/extracted_data")
    org_midi_path = Path("./streamlit_data/original_data")
    midi_path = Path("./streamlit_data/midi_files")
    mp3_files = get_files(mp3_path)
    midi_files = get_files(midi_path)

    file_select = [os.path.basename(x) for x in mp3_files]
    file = st.sidebar.selectbox("Choose file", file_select)
    mp3_file = mp3_path/file

    # process original files
    org_file = str(org_midi_path) + "/" + file

    #midi_file
    mid_file = file[:-4] + '.mid'
    midi_file = midi_path/mid_file

    return midi_file, org_file, mp3_file, 1
    #vocab


def raw_midi(midi_file):
    
    st.subheader("Raw output of midi file converter")   
    stream =  converter.parse(midi_file)
    if len(stream.recurse()) >= 5:
        slider_note_visualize = st.slider("Choose number of notes to see:",0, int(len(stream.recurse())/5), 1)
        iterate = 0
        for s in stream.recurse():
            if iterate == slider_note_visualize:
                break
            st.write(s)
            iterate += 1
    else:
        st.write("Converter gave an Empty Stream")

def listToString(s): 
    
    # initialize an empty string
    str1 = " " 
    
    # return string  
    return (str1.join(s))

def tokenize_midi(midi_file):

    st.subheader("Tokenized output of the midi file")

    from miditok import REMI, get_midi_programs
    from miditoolkit import MidiFile

    # Our parameters
    pitch_range = range(21, 109)
    beat_res = {(0, 4): 8, (4, 12): 4}
    nb_velocities = 32
   
    # Creates the tokenizer and loads a MIDI
    tokenizer = REMI(pitch_range, beat_res, nb_velocities, mask=True)
    midi1= "streamlit_data\midi_files\Zazen_Mix.mid"
    midi = MidiFile('streamlit_data\midi_files\Zazen_Mix.mid')
    #path/to/your_midi.mid
    # Converts MIDI to tokens, and back to a MIDI
    tokens = tokenizer.midi_to_tokens(midi)
    converted_back_midi = tokenizer.tokens_to_midi(tokens, get_midi_programs(midi))

    # Converts just a selected track
    tokenizer.current_midi_metadata = {'time_division': midi.ticks_per_beat, 'tempo_changes': midi.tempo_changes}
    piano_tokens = tokenizer.track_to_tokens(midi.instruments[0])
    st.write("The track is: ")
    st.audio(midi1, start_time=0)
    # And convert it back (the last arg stands for (program number, is drum))
    converted_back_track, tempo_changes = tokenizer.tokens_to_track(piano_tokens, midi.ticks_per_beat, (0, False))
    st.write("The tokenized track is:")
    st.markdown("##")
    #st.write(listToString(piano_tokens))
    st.write(piano_tokens)
    #print(type(piano_tokens))
    """try:
        item = MusicItem.from_file(midi_file, vocab)
        slider_token_len = st.slider("choose length of tokens you want to see",0, len(item.to_text()), int(len(item.to_text())/5))
        st.write( item.to_text()[:slider_token_len]+ " ...")
    except:
        st.write("Exception raised during handling {} ".format(midi_file))
        st.write("Please select another file from sidebar")  """

def music_midi(mp3_ex, org_file):

    st.info("Allows to listen to both the original version of the MIDI we used for \
                    extraction and also the extracted version of the melody.")

    st.subheader("Play MIDI")
    st.text("Choose File from the sidebar to listen to a sample")

    st.subheader("Play Original MIDI")
    play_updated(org_file)

    st.write("\n")
    st.subheader("Play Piano extracted MIDI")
    play_updated(mp3_ex)


def load_pred_data(name="pred_lmd"):
    
    trim_pred_dir = Path("./streamlit_data")/name/Path("full")
    input_file_org = Path("./streamlit_data")/name/Path("input/original")
    input_file_trim = Path("./streamlit_data")/name/Path("input/trimmed")
    pred_file = Path("./streamlit_data")/name/Path("predicted")

    trim_pred_files = get_files(trim_pred_dir/"midi")
    input_org_files = get_files(input_file_org/"midi")
    input_trimo_files = get_files(input_file_trim/"midi")
    pred_files = get_files(pred_file/"midi")

    file_select = [os.path.basename(x) for x in trim_pred_files]
    file = st.sidebar.selectbox("Choose file", file_select)

    # music files
    trim_pred_disp = trim_pred_dir/"midi"/file
    input_org_disp = input_file_org/"midi"/file
    input_trim_disp = input_file_trim/"midi"/file
    pred_disp = pred_file/"midi"/file

    # notes
    filename = file.replace(".mp3", "")
    filename = filename + ".mid-1" + ".png"

    trim_pred_note = trim_pred_dir/"notes"/filename
    input_org_note = input_file_org/"notes"/filename
    input_trim_note = input_file_trim/"notes"/filename
    pred_note = pred_file/"notes"/filename

    return [input_org_disp, input_trim_disp, pred_disp, trim_pred_disp] , [input_org_note, input_trim_note, pred_note, trim_pred_note]

def play_pred(name="pred_lmd"):
    st.subheader("Users have four files they can listen to.")
    st.markdown("* **Original input file** : This file is the original full length music file in the \
                test dataset.")
    st.markdown("* **Trimmed input file** : This file is the trimmed/clipped part of the original file \
                which is used by the inference engine to predict the next notes.")
    st.markdown("* **The prediction** : This is the file that is the prediction produced by the \
                inference engine.")
    st.markdown("* **Trimmed Input + prediction file** : Final file is the augmentation of the clipped \
                file and the prediction of the model.")

    music_files, notes = load_pred_data(name)

    select_action = st.sidebar.selectbox("Action", ["Play Predictions", "Note charts"])

    if select_action == "Play Predictions":
        st.subheader("Choose any file from the sidebar to listen to a sample")

        st.subheader("Play Original File")
        play_updated(music_files[0])
        
        st.subheader("Play Trimmed input")
        play_updated(music_files[1])
        
        st.subheader("Play Prediction")
        play_updated(music_files[2])

        st.subheader("Play Trimmed + Prediction")
        play_updated(music_files[3])

    else:
        st.subheader("Choose any file from the sidebar to see note charts comparison")

        select_chart = st.radio("Choose note chart to show", ["Original", "Trimmed", "Prediction", "Trimmed + Prediction"])

        if select_chart == "Original":
            st.subheader("Original File")
            st.image(str(notes[0]), use_column_width=True)
        elif select_chart == "Trimmed":
            st.subheader("Trimmed input")
            st.image(str(notes[1]), use_column_width=True)
        elif select_chart == "Prediction":
            st.subheader("Prediction")
            st.image(str(notes[2]), use_column_width=True)
        elif select_chart == "Trimmed + Prediction":
            st.subheader("Trimmed + Prediction")
            st.image(str(notes[3]), use_column_width=True)




def loadd_pred_data(name="pred_midi"):
    
    trim_pred_dir = Path("./streamlit_data")/name/Path("full")
    input_file_org = Path("./streamlit_data")/name/Path("input/original")
    input_file_trim = Path("./streamlit_data")/name/Path("input/trimmed")
    pred_file = Path("./streamlit_data")/name/Path("predicted")

    trim_pred_files = get_files(trim_pred_dir/"midi")
    input_org_files = get_files(input_file_org/"midi")
    input_trimo_files = get_files(input_file_trim/"midi")
    pred_files = get_files(pred_file/"midi")

    file_select = [os.path.basename(x) for x in trim_pred_files]
    file = st.sidebar.selectbox("Choose file", file_select)

    # music files
    trim_pred_disp = trim_pred_dir/"midi"/file
    input_org_disp = input_file_org/"midi"/file
    input_trim_disp = input_file_trim/"midi"/file
    pred_disp = pred_file/"midi"/file

    # notes
    filename = file.replace(".mp3", "")
    filename = filename + ".mid-1" + ".png"

    trim_pred_note = trim_pred_dir/"notes"/filename
    input_org_note = input_file_org/"notes"/filename
    input_trim_note = input_file_trim/"notes"/filename
    pred_note = pred_file/"notes"/filename

    return [input_org_disp, input_trim_disp, pred_disp, trim_pred_disp] , [input_org_note, input_trim_note, pred_note, trim_pred_note]

def play_predd(name="pred_midi"):
    st.subheader("Users have four files they can listen to.")
    st.markdown("* **Original input file** : This file is the original full length music file in the \
                test dataset.")
    st.markdown("* **Trimmed input file** : This file is the trimmed/clipped part of the original file \
                which is used by the inference engine to predict the next notes.")
    st.markdown("* **The prediction** : This is the file that is the prediction produced by the \
                inference engine.")
    st.markdown("* **Trimmed Input + prediction file** : Final file is the augmentation of the clipped \
                file and the prediction of the model.")

    music_files, notes = loadd_pred_data(name)

    select_action = st.sidebar.selectbox("Action", ["Play Predictions", "Note charts"])

    if select_action == "Play Predictions":
        st.subheader("Choose any file from the sidebar to listen to a sample")

        st.subheader("Play Original File")
        play_updated(music_files[0])
        
        st.subheader("Play Trimmed input")
        play_updated(music_files[1])
        
        st.subheader("Play Prediction")
        play_updated(music_files[2])

        st.subheader("Play Trimmed + Prediction")
        play_updated(music_files[3])

    else:
        st.subheader("Choose any file from the sidebar to see note charts comparison")

        select_chart = st.radio("Choose note chart to show", ["Original", "Trimmed", "Prediction", "Trimmed + Prediction"])

        if select_chart == "Original":
            st.subheader("Original File")
            st.image(str(notes[0]), use_column_width=True)
        elif select_chart == "Trimmed":
            st.subheader("Trimmed input")
            st.image(str(notes[1]), use_column_width=True)
        elif select_chart == "Prediction":
            st.subheader("Prediction")
            st.image(str(notes[2]), use_column_width=True)
        elif select_chart == "Trimmed + Prediction":
            st.subheader("Trimmed + Prediction")
            st.image(str(notes[3]), use_column_width=True)

def intro():
    select = st.sidebar.selectbox("Options", ["GUI Info", "Outline"])
    if select == "GUI Info":
        print_gui_info()
    else:
        print_outline()


def perfparam():
    select = st.sidebar.selectbox("Options", ["General", "Parameters", "Visualization"])
    if select == "General":
        print_pparameters()
    elif select == "Parameters":
        print_paramters()
    else:
        print_visualize()


def print_paramters():
    st.markdown("***")
    st.header("Paramters taken into account in the project:")
    st.write("##")
    st.subheader("Amplitude Envelope")
    st.write("The Amplitude Envelope (AE) aims to extract the maximum amplitude within each frame and string them all together. It is important to remember that the amplitude represents the volume (or loudness) of the signal.")
    st.write("we split up the signal into its constituent windows and find the maximum amplitude within each window. \
             From there, we plot the maximum amplitude in each window along time.\
             We can use the AE for onset detection, or the detection of the beginning of a sound.")
    st.markdown("***")
    st.subheader("Root-Mean-Square Energy")
    st.write("It attempts to perceive loudness, which can be used for event detection. Furthermore, it is much more robust against outliers, meaning if we segment audio, we can detect new events (such a a new instrument, someone speaking, etc.) much more reliably.")
    st.write("We square the amplitudes within the window and sum them up. Once that is complete, we will divide by the frame length, take the square root, and that will be the RMS energy of that window.")
    st.write("To extract the RMS, we can simply use librosa.feature.rms")
    st.markdown("***")
    st.subheader("Zero Crossing Rate")
    st.write("The Zero-Crossing Rate (ZCR) aims to study the the rate in which a signal’s amplitude changes sign within each frame. Compared to the previous two features, this one is quite simple to extract.")
    st.write("This feature is relevant for identifying percussion sound as they often have fluctuating signals that can ZCR can detect quite well as well as pitch detection. However, this feature is generally used as a feature in speech recognition for voice activity detection.")
    st.write("Using librosa, we can extract the ZCR using librosa.feature.zero_crossing_rate")
    st.write("***")

def print_visualize():
    st.write("***")
    st.header("Visualization of the parameters taken into account")
    st.markdown("##")
    st.subheader("Select the Song whose Performance Parameters is to be checked:")
    st.markdown(" Songs are from the Midi Dataset")
    st.markdown("***")
    songs = ("Cosmo", "Eternal Harvest", " Eyes On Me Piano", "Great War", "Vincent Piano")
    selected_song = st.selectbox(" Select the song", songs)
    songs1 =''

    if selected_song == "Cosmo":
        st.markdown("##")
        st.subheader("1. Wave Form of Input song vs Generated song")
        st.markdown("##")

        st.image("./images/cosmos_wf.png" , width = 850)
        #st.image("./images/magenta.png", use_column_width=True)
        st.markdown("***")

        st.subheader("2. ZCR Energy")
        #st.markdown("##")
        st.image("./images/cosmos_zcr.png" ,width = 850)
        st.markdown("***")

        st.subheader("3. STFT chromagrams ")
        st.markdown("##")
        st.image("./images/cosmos_chroma.png" , width = 700)
        st.markdown("##")
        st.markdown("***")
    
    elif selected_song == "Eternal Harvest":
        st.markdown("##")
        st.subheader("1. Wave Form of Input song vs Generated song")
        #st.image
        st.markdown("##")
        st.image("./images/eternal_wf.png" , width = 850)
        st.markdown("***")

        st.subheader("2. ZCR Energy")
        #st.image
        st.image("./images/eternal_zcr.png" , width = 850)
        st.markdown("***")

        st.subheader("3. STFT chromagrams ")
        #st.image
        st.image("./images/eternal_chroma.png" , width = 700)
        st.markdown("##")
        st.markdown("***")
    
    elif selected_song == "Eyes On Me Piano":
        st.markdown("##")
        st.subheader("1. Wave Form of Input song vs Generated song")
        #st.image
        st.markdown("##")
        st.image("./images/eyesonme_wf.png" , width = 850)
        st.markdown("***")

        st.subheader("2. ZCR Energy")
        st.image("./images/eyesonme_zcr.png" , width = 850)
        st.markdown("***")

        st.subheader("3. STFT chromagrams ")
        st.image("./images/eyesonme_chroma.png" , width = 700)

        st.markdown("##")
        st.markdown("***")
    
    elif selected_song == "Great War":
        st.markdown("##")
        st.subheader("1. Wave Form of Input song vs Generated song")
        st.markdown("##")
        st.image("./images/greatwar_wf.png" , width = 850)
        st.markdown("***")

        st.subheader("2. ZCR Energy")
        st.image("./images/greatwar_zcr.png" , width = 850)
        st.markdown("***")

        st.subheader("3. STFT chromagrams ")
        st.image("./images/greatwar_chroma.png" , width = 700)

        st.markdown("##")
        st.markdown("***")
    
    else:

        st.markdown("##")
        st.subheader("1. Wave Form of Input song vs Generated song")
        st.markdown("##")
        st.image("./images/vincent_wf.png" , width = 850)
        st.markdown("***")

        st.subheader("2. ZCR Energy")
        st.image("./images/vincent_zcr.png" , width = 850)
        st.markdown("***")

        st.subheader("3. STFT chromagrams ")
        st.image("./images/vincent_chroma.png" , width = 700)
        st.markdown("##")
        st.markdown("***")
 

def predictions():
	st.subheader("Under Construction..")
	pass