import streamlit as st
#import graphviz as graphviz
def print_intro():
	st.title("Automated Music Generation using Deep Learning Techniques")
	st.subheader("Final Year Project - Batch B14")
	st.info(" To switch in between different functionalities, please choose from the sidebar. ")
	st.write("##")
	st.write("By Akash Shankar, Ankith Praveen, Guru Dutt, N Rafaath")
	#st.write('Many practices have been introduced in music age as of late. While elaborate music age utilizing profound learning procedures has turned into the standard, these models actually battle to create music with high musicality, various degrees of music construction, and controllability.')
	#st.write('Furthermore, more application situations, for example, music treatment require mimicking additional particular melodic styles from a couple of given music models, rather than catching the general sort style of a huge information corpus.')
	st.write("***")
	
	#st.markdown('##')
	#st.markdown('The notion of using Deep Learning techniques to generate musical sequences in ABC notation is explored in this study.')
	#st.markdown('The suggested method takes the Nottingham dataset and enciphers the ABC notations as input to neural networks. The main goal is to feed an arbitrary note into neural networks')
	#st.write("##")
	#st.markdown('Then let them analyze and reshape a sequence built on the note until a good fragment of music is generated. Multiple alterations have been done to optimize networks parameters for ideal generation. \
				#Through a deep hierarchical structure, the music output piece is checked on metrics of harmony, grammatical correctness and rhythm directly from raw input data ')
	

def print_choose():
	st.info("Data Analysis dives deeper into the Data Preprocessing and Tokenization pipeline to \
			give a better visualization about the whole process. This section will try to summarise \
			how the data is taken in its RAW form and change into model ready format.")

	st.markdown("Choose the approporiate\
				action from the 'Select Function' on the sidebar")


def print_outline():
	st.title("Outline")
	# GOAL
	st.subheader("Goal:")
	st.markdown("Generate monophonic music using deep learning")
	st.subheader("Revised proposal: ")
	st.markdown("Use deep learning techniques to create a system that can predict/create \
				 just piano (single instrument) music, given input an clipped music file.")
	st.subheader("Data:")
	st.markdown("Data collection/processing phase took about ~2 weeks. ")
	st.markdown("During Phase 1, our use case for a single instrument music generation included extracting midi melodies from all the music sources from (Namely ABC Nottingham dataset notations and the Google Magneta datasets, since the amount \
				 of data we had for just piano was not enough to train a model like Elowsson and Friberg, Variational Auto-Encoders or TRANSFORMERXL\
				 from scratch.")

	st.markdown("During Phase 2, we used midi file sources without any extraction processes, since the \
				data preprocessing pipeline for model training took care of that in the code.")
	# PHASES
	st.subheader("Phases:")
	st.markdown("Our project, like every other group project evolved during the course. Our initial specifications \
				included using the extracted piano midi files, for music generation \
				but after running into problems which delayed our progress, we started to work in parallel \
				with other model architectures and methodolgies to keep the pace of our project intact. The key \
				differences between Phase 1 & Phase 2 are: ")
	st.markdown("1. Model Architecture")
	st.markdown("2. Use of extracted piano files in Phase 1 for the models")
	# Predictions
	st.subheader("Predictions:")
	st.markdown("We generate predictions by clipping a testing file and letting the trained model\
				 predict the rest of the music part by itself.")
def print_gui_info():
	# SECTIONS
	st.image("./images/intro.jpg", width=800)
	st.markdown('The notion of using Deep Learning techniques to generate musical sequences in ABC notation is explored in this study.')
	st.markdown('The suggested method takes the Nottingham dataset and enciphers the ABC notations as input to neural networks. The main goal is to feed an arbitrary note into neural networks')
	#st.write("##")
	st.markdown('Then let them analyze and reshape a sequence built on the note until a good fragment of music is generated. Multiple alterations have been done to optimize networks parameters for ideal generation. \
				Through a deep hierarchical structure, the music output piece is checked on metrics of harmony, grammatical correctness and rhythm directly from raw input data ')
	st.write("***")
	st.header("Graphical User Interface Information")
# Introduction
	st.write("##")
	st.subheader("Section 1: Introduction")
	#st.write("##")
	st.markdown("This section is the welcoming interface for the project. This section will help \
				explain what the GUI is all about, the different sections the project has, how are they divided \
				, so that navigation can be done easily and without hindrance.")
	st.markdown("On the sidebar you have the option to choose between three other sub-sections")
	st.markdown("1. GUI Information" )
	st.markdown("2. Outline ")
	st.markdown("3. Literature Review")
	st.subheader("GUI Information ")
	st.markdown("This is the page where the user will land by defualt, which contains all the \
				 information about the GUI for guidance, divided into its subsections by the sidebar.")
	st.subheader("Outline")
	st.markdown("This section will provide a brief overview of how the project was approached and its outline.")

	st.subheader("Literature Review")
	st.markdown("This sub section will include the existing literature and methodologies that have been \
				developed and researched for producing music through deep learning.")
# Data Analysis
	st.write("***")
	st.subheader("Section 2: Data Analysis")
	st.markdown("This section dives deeper into the Data Processing and tries to \
				take a visual/audio eplanatory approach. Subsections include ")
	st.markdown("1. Data Info")
	st.markdown("2. Raw MIDI")
	st.markdown("3. Tokenized MIDI")
	st.markdown("4. Play MIDI")

	st.subheader("Data Info")
	st.markdown("Shows the user visual representation of the data collected and the sources used \
				for the project")

	st.subheader("RAW MIDI")
	st.markdown("Shows the RAW version of the MIDI file that we choose from the sidebar. Moreover, \
				briefly explains about the music21 library which was used for data processing.")
	
	st.subheader("Tokenized MIDI")
	st.markdown("Shows tokenized version of our midi files and also explains the process of tokenization which is widely used in the Natural Language Processing ")

	st.subheader("Play MIDI")
	st.markdown("Uses the loaded files and plays for the user the original MIDI file and the piano extraction \
				of the MIDI file generated by our model")
    
	st.write("***")
	st.subheader("Section 3: Model Description")
	st.markdown("This section will briefly describe about the models and the phases of our training pipeline using \
				different architectures and methodologies, which is divided into Phase I and Phase II.")

	# Prediction
	st.write("***")
	st.subheader("Section 4: Predictions")
	st.markdown("Predictions will deal with the output from our trained models. It will have an original file, clipped\
				predicted file and the trimmed + predicted file, which the user can play and visualize.")
	st.markdown("There are two trained models, one is the Reddit Pop Model, which is trained on pop music, and the second is \
				the Maestro Dataset model, which is trained on dataset of classical songs.")

def print_lit_review():
	st.title("Literature Review")
	st.markdown("This section will go through the literature review our group did for the project. It will also discuss \
			the state of the art technologies that have already been used for music generation using AI.")
	
	# Magenta
	st.markdown("## Magenta (Google Brain)")
	st.markdown("Prior Work:")
	st.markdown("* Vanilla approaches include training a RNN (LSTM) model to predict the next note in a musical sequence (e.g. Eck and Schmidhuber 2002).")
	st.markdown("* Similar to character RNN, these Note RNNs were used to generate melodies by initializing them with a short sequence, and then obtaining next notes from the model by repeatedly sampling from the model’s output")
	st.markdown("Problems:")
	st.markdown("* Excessively repeating tokens (less creativity)")
	st.markdown("* Producing sequences that lack a consistent theme or structure (straying from music rules and structure)")
	st.markdown("* Wandering and random sequences (randomness)")
	st.markdown("Research Question:)")
	st.markdown("* Given music has relatively well-defined structural rules, can a simple Note RNN maintain that structural integrity?")	
	st.markdown("Proposition")
	st.markdown("* Given trained Note RNN, goal is to teach it concepts about music theory, while maintaining the information about typical melodies originally learned from data.")
	st.markdown("RL Tuner Design:")
	st.markdown("* Three networks: [The Q network, The Target-Q network] (Deep Q-Learning) and a Reward RNN")
	st.markdown("* Q-network, Target Q-network: recurrent LSTM model, architecture same as Note RNN")
	st.markdown("* Reward RNN: used to supply part of the reward value used to train model. Held fixed during training.")
	st.image("./images/magenta.png", use_column_width=True)


	# MuseGAN
	st.markdown("## MuseGAN")

	st.markdown("High Level Idea:")
	st.markdown("* As the name suggests, this strategy uses Generative Adversarial Networks (GANs)")
	st.markdown("* Three models for symbolic multi-track music generation")
	st.markdown("* The jamming model, the composer model and the hybrid model")
	st.markdown("* The paper shows that the models can generate coherent music of four bars right from scratch")
	st.markdown("Challenges:")
	st.markdown("* Have an account for the hierarchical, temporal, and the structural patterns of music")
	st.markdown("* Musical notes are often grouped into chords, arpeggios, or melodies, so chronologically ordering of notes is not suitable")
	st.markdown("Goal:")
	st.markdown("* Generate multi-track polyphonic music with harmonic and rhythmic structure, multitrack interdependency, temporal structure")
	st.markdown("Data Representation:")
	st.markdown("* They use multiple-track piano-roll representation")
	st.markdown("* The piano-roll dataset used is derived from the Lakh MIDI dataset (LMD) (Raffel 2016),4 a large collection of 176,581 unique MIDI files. The MIDI files are converted to multi-track piano-roll.")
	st.image("./images/MuseGAN.png", use_column_width=True)

	# Musenet (Open AI)
	st.markdown("## Musenet (Open AI)")
	st.markdown("The model:")
	st.markdown("* A transformer based model that can generate 4-minute musical compositions with 10 different instruments, and can combine styles from country to Mozart to the Beatles.")

	st.markdown("Approach:")
	st.markdown("* Uses the same general-purpose unsupervised technology as GPT-2, a large-scale transformer model trained to predict the next token in a sequence, whether audio or text.")
	st.markdown("Open AI GPT2:")
	st.markdown("* GPT-2 (2nd version of Open AI GPT) is a large transformer-based language model with 1.5 billion parameters, \
				trained on a dataset of 8 million web pages. GPT-2 is trained with a simple objective: predict the next word, \
				given all of the previous words within some text.")
	st.markdown("Limitations:")
	st.markdown("* Computation time and cost, considering the number of parameters mentioned above.")
	st.markdown("* The instruments you ask for are strong suggestions, not requirements. MuseNet generates each note by calculating the probabilities across all possible notes and instruments.")
	st.markdown("* The model shifts to make your instrument choices more likely, but there’s always a chance it will choose something else.")
	st.markdown("* MuseNet has a more difficult time with odd pairings of styles and instruments (such as Chopin with bass and drums). Generations will be more natural if you pick instruments closest to the composer or band’s usual style.")

	# WaveNet
	st.markdown("## WaveNet")
	st.markdown("What it does:")
	st.markdown("* WaveNet is a CNN based autoregressive  and fully probabilistic generative model that directly models the raw waveform of \
				the audio signal(i.e music or human speech), one sample at a time.")
	st.markdown("* It uses Gated PixelCNN architecture, dilated convolutions and causal convolutions.")
	st.markdown("Original purpose:")
	st.markdown("* To create human like speech better than the Text To Speech models, to make the output more close to real time human speech.")
	st.markdown("Implementation:")
	st.markdown("* The model is a CNN, where the convolutional layers have multiple dilatation factors and predictions only depend on \
				previous timesteps. i.e predictions only depends upto ‘t’ and not ‘t+1,t+2,...t+n’")
	st.markdown("* The Residual and Skip connections help the raw audio directly impact the output.")
	st.markdown("Results:")
	st.markdown("* In context of the speech generation, at each step during sampling a value is drawn from the probability distribution computed by the network and is \
				fed back into the input and a new prediction for the next step is made , for realistic-sounding audio.")
	st.markdown("* Further, the model was also used on piano dataset and new samples were generated opening possibilities for music generation.")
def print_phase1():
	st.header("Data Preprocessing and extraction")
	st.subheader("Model followed for data preprocessing and extraction")
	st.image("./images/ph1.jpeg", width=500)#Insert the preprocesssing pictures from our report
	st.markdown("## Integer Encoding")
	st.markdown("This is the interpretation of musical notes into unique characters for the respective identification of each note.")
	st.markdown("## One Hot Encoding")
	st.markdown("This creates a new binary vector for each possible label which is later fed into the LSTM units in batches.")
	
	st.subheader("Used Data")
	st.markdown("We used only Piano data after running the extraction due to")
	st.markdown("* Availability of data")
	st.markdown("* Non Availability of resources")

def print_algo():
	st.title("Algorithm steps of the Project: ")
	st.markdown("##")
	st.subheader("1. Importing Basic libraries ")
	st.markdown("##")
	st.write(" Basic libraries are imported: ")
	st.markdown("* Music21")
	st.markdown("* Numpy")
	st.markdown("* Pandas")
	st.markdown("* FluidSynth")
	st.markdown("***")

	st.subheader("2. Reading and Parsing the Midi File ")
	st.markdown("##")
	st.write(" For this project, we will be only working on files that contain sequential streams of Piano data. We will separate all files by their instruments and use only Piano. \
			   Piano stream from the midi file contains many datas  like Keys, Time Signature, Chord, Note etc. We don’t require all of this except Notes and Chords to generate music. \
			   Lastly, we will return arrays of notes and chords. ")
	st.markdown("***")

	st.subheader("3. Exploring the dataset ")
	st.write("For this project we will be going to use 50 as a threshold frequency. So we will take only those notes which have frequencies more than 50.\
		      You can anytime change these parameters.")
	st.write("Also, we will change our ‘notes_array’ which will contain notes that are greater than threshold frequency.")
	st.write(" Example - filter notes greater than threshold i.e. 50\
			   freq_notes=dict(filter(lambda x:x[1]>=50,freq.items())) ")
	st.markdown("***")

	st.subheader("4. Input and Output Sequence for model ")
	st.markdown("***")

	st.subheader("5. Training and Testing sets ")
	st.markdown("We will reshape our array for our model and split the data into 80:20 ratio. 80% for the training set and 20% for the testing set.")
	st.markdown("***")

	st.subheader("6. Building the model")
	st.write("Going to use LSTM model architecture. We will use 2 stacked LSTM layers with a dropout rate of 0.2. Dropout basically prevents overfitting while training the model, while it does not affect the inference model. \
			 Finally we will be using a fully connected Dense layer for output. ")
	st.write("Output dimension of the Dense Layer will be equal to the length of our unique notes along with the ‘softmax’ activation function which is used for multi-class classification problems.")
	st.markdown("***")

	st.subheader("7. Model Training")
	st.write(" After building the model, we will now train it on the input and output data. For this will be using ‘Adam’ optimizer on batch size of 128 and for total 80 epochs. \
			   After we finish training, save the model for prediction. ")
	st.markdown("***")

	st.subheader("8. Inference and Saving the file")
	st.write("Using the trained model we will predict the notes ")
	st.write("Firstly generate a random integer(index) for our testing input array which will be our testing input pattern. We will reshape our array and predict the output. ")		 
	st.write("Using the ‘np.argmax()’ function, we will get the data of the maximum probability value. Convert this predicted index to notes using ‘ind2note’(index to note) dictionary")
	st.write("With the predicted output notes, Now we will save them into a MIDI file.")


def print_phase2():
	st.title("Training and deploying the model")
	st.markdown("##")
	col1, col2, col3 = st.columns(3)
	with col1:
		st.write(' ')
	with col2:
		st.image("./images/ph2.jpeg", width=250)
	with col3:
		st.write(' ')
	st.markdown("## Main Takeaways")
	st.markdown("The model built on 3 LSTM layers, which acts as the core model. Various Dropout layers are added to prevent the over fitting of the data. ")
	st.markdown("To calculate the loss for each iteration of the training we will be using categorical cross entropy since each of our outputs only belongs to a single class and we have more than two classes to work with. And to optimize our network we will use a Adam optimizer as it is usually a very good choice for recurrent neural networks.")
	st.markdown("LSTM layers is a Recurrent Neural Net layer that takes a sequence as an input and can return either sequences (return sequences=True) or a matrix. \
Dropout layers are a regularization technique that consists of setting a fraction of input units to 0 at each update during the training to prevent over fitting. The fraction is determined by the parameter used with the layer.")
	st.markdown("***")
	st.markdown("The model.fit() function in Keras is used to train the network.\
				The first sequence we submit is the sequence of notes at the starting index. \
				For every subsequent sequence that we use as input, we will remove the first note of the sequence and insert the output of the previous iteration at the end of the sequence.")
	col2a, col3a = st.columns(2)
	#with col1a:
		#st.write(' ')
	with col2a:
		st.image("./images/ph2a.jpg", width=500)
	with col3a:
		st.write(' ')
	
	
	st.markdown("##")
	st.markdown("To determine the most likely prediction from the output from the network, we extract the index of the highest value. \
				The value at index X in the output array corresponds to the probability that X is the next note.")
	col2b, col3b = st.columns(2)
	with col2b:
		st.image("./images/ph2b.jpg", width=550)
	with col3b:
		st.write(' ')
	
	st.markdown("***")
	
	st.markdown("## Training time.")
	st.markdown("Maestro - 400000 epochs - 8 days")
	st.markdown("Magneta -450000 epochs -9 days") 

def print_pparameters():

	st.markdown("***")
	st.write("Music exists only with performance ")
	st.write(" \t • performance realizes acoustic rendition of musical ideas")
	st.write("• each rendition is unique ")
	st.write("• score information is interpreted, modified, added to, or dismissed ")
	st.write("• adds “expressivity” ")
	st.markdown("##")
	st.image("./images/pparam.jpg", width=800)
	st.markdown("***")
	st.subheader("By analyzing the music performance, we learn about: ")
	st.markdown("##")
	st.subheader("The Performance:")
	st.write("− General performance characteristics \
			  − Notable stylistic differences (over time, between artists, …)")
	st.subheader("The Performer")
	st.write(" − Mapping of intent and projected emotion to measurable parameters")
	st.subheader("The Listener")
	st.write("− what is perceived as (appropriate level of) expressiveness ")
	st.write("− how can different performance parameters impact the listener ")
	st.write("− How is aesthetic perception shaped by performance parameters")
	st.markdown("***")

	st.subheader("Perceptual relevance of “expressive” performance characteristics:")
	st.write("• dynamics highest impact on ratings of emotional expression ")
	st.write(" • expressive timing best predicts ratings of musical tension ")
	st.write(" • sharpened intonation at phrase climax contributes to perceived excitement")
	st.markdown("***")

	st.subheader(" Measured ≠ Perceived")
	st.write("e.g., measurable difference between “normative” and “expressive” performance does not necessarily lead to \
			perception of expressivity")
	st.write("• e.g., no correlation between measured and perceived vibrato onsets")