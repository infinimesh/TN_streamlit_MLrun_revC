## -------------------------------
## ====    Import smthng      ====
## -------------------------------

##Libraries for Streamlit
##--------------------------------
import streamlit as st
import io
from scipy.io import wavfile as scipy_wav
from PIL import Image

##Libraries for prediction
##--------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models





## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


## Page decorations
##--------------------------------


id_logo = Image.open("TypoMeshDarkFullat3x.png")
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image(id_logo)

st.markdown("<h1 style='text-align: center; color: grey;'>ML Audio Recognition App</h1>", 
            unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Truck Norris</h1>", 
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>Canvas Cutting Detection</h3>", 
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: black;'>App for comparison ML models</h3>", 
            unsafe_allow_html=True)
st.header(" ")
st.header(" ")







## -------------------------------
## ====  Select and load data ====
## -------------------------------
# st.header("Select data to analyze")
st.markdown("<h2 style='text-align: center; color: grey;'>Select data to analyze</h2>", 
            unsafe_allow_html=True)


st.subheader("Select one of the samples")

selected_provided_file = st.selectbox(label="", 
                            options=["example of a cutting event", "example of a background sound"]
                            )


st.subheader("or Upload an audio file in WAV format")
st.write("if a file is uploaded, previously selected samples are not taken into account")

uploaded_audio_file = st.file_uploader(label="Select a short WAV file < 5 sec", 
                                        type="wav", 
                                        accept_multiple_files=False, 
                                        key=None, 
                                        help=None, 
                                        on_change=None, 
                                        args=None, 
                                        kwargs=None, 
                                        disabled=False)




uploaded_audio_file_4debug = uploaded_audio_file #this expression must be before Data Switch

## Data Switch is here
##--------------------------------
if uploaded_audio_file is not None:
    # st.write("YEP")
    audio_arr_sr, audio_arr = scipy_wav.read(uploaded_audio_file)
else:
    # st.write("NOPE") #444debug
    if selected_provided_file == "example of a cutting event":
        audio_arr_sr, audio_arr = scipy_wav.read('03-CM01B_Vorne.wav')
    if selected_provided_file == "example of a background sound":
        audio_arr_sr, audio_arr = scipy_wav.read('04-Schlitzen_am_LKW.wav')

## If stereo, then do averaging over channels 
##-------------------------------------------
if len(audio_arr.shape) > 1:
    audio_arr = np.mean(audio_arr, axis=1, dtype=int)

## Normalize values of audio 
##--------------------------
audio_arr = audio_arr / np.max(audio_arr)


## Convert to virtula file to play it 
##-------------------------------------------
virtualfile = io.BytesIO()
scipy_wav.write(virtualfile, rate=audio_arr_sr, data=audio_arr)
uploaded_audio_file = virtualfile





## -------------------------------
## ====   Show selected data  ====
## -------------------------------

# st.subheader("Show the data selected for analysis")
st.header(" ")
st.header(" ")
st.markdown("<h2 style='text-align: center; color: grey;'>Show the data selected for analysis</h2>", 
            unsafe_allow_html=True)

# st.write("Listen the loaded data")
st.markdown(" ##### _Listen the loaded data_")
st.audio(uploaded_audio_file, format='audio/wav')
# st.write("Waveform of the loaded data")
st.markdown(" ##### _Waveform of the loaded data_")
# st.line_chart(audio_arr) #commented bcoz it is not so convenient to use on the page
fig_wf, ax_wf = plt.subplots(1,1, figsize=(5, 2))
ax_wf.plot(audio_arr)
ax_wf.grid('True')
st.pyplot(fig_wf)


# ----------------------------------------
# ==== Functions to make spectrograms ====
# ----------------------------------------

def get_spectrogram( waveform, sampling_rate ):
    waveform_1d = tf.squeeze(waveform)
    waveform_1d_shape = tf.shape(waveform_1d)
    n_samples  = waveform_1d_shape[0]
    spectrogram = tf.signal.stft(
                        tf.squeeze(tf.cast(waveform, tf.float32)),
                        frame_length=tf.cast(n_samples/100, dtype=tf.int32),
                        frame_step=tf.cast(n_samples/100/4, dtype=tf.int32),
                        )
    spectrogram = tf.abs(spectrogram)
    l2m = tf.signal.linear_to_mel_weight_matrix(
                        num_mel_bins=125,
                        num_spectrogram_bins=tf.shape(spectrogram)[1],
                        sample_rate=sampling_rate,
                        lower_edge_hertz=0,
                        upper_edge_hertz=22000,
                        )
    spectrogram = tf.matmul(spectrogram, l2m)
    spectrogram = tf.math.divide(spectrogram, tf.math.reduce_max(spectrogram) )
    spectrogram = tf.math.add(spectrogram, tf.math.reduce_min(spectrogram) )
    spectrogram = tf.math.add(spectrogram, 0.01 )
    spectrogram = tf.math.log(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.transpose(spectrogram, perm=(1,0,2))
    spectrogram = spectrogram[::-1, :, :]
    return spectrogram


#default values which is used to plot spectrogram and as input_shape unless it is changed according to a model specs
#once again! this value might be overwritten for a specific model
spectrogram_shape_to_analyze = (64*2*1, 64*4*1)


def spectrogram_resize(spectrogram):
    return tf.image.resize(spectrogram, spectrogram_shape_to_analyze)



# ----------------------------------------
# ==== Create Dataset of spectrograms ====
# ----------------------------------------


spectrogram_arr = get_spectrogram(audio_arr, audio_arr_sr)
spectrogram_arr_resized = spectrogram_resize(spectrogram_arr)


## Show spectrogram
##--------------------------------
st.markdown(" ##### _Spectrogram of the  loaded data_")
fig_sp, ax_sp = plt.subplots(1,1, figsize=(5, 2))
ax_sp.imshow(spectrogram_arr_resized)
st.pyplot(fig_sp)




## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
## >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



## -------------------------------
## ====    Apply ML model     ====
## -------------------------------

st.header(" ")
st.header(" ")
st.markdown("<h2 style='text-align: center; color: grey;'>Analysis with ML model</h2>", 
            unsafe_allow_html=True)




"In the followong list one can find"
st.subheader("Two revisions of the ML models:")
"rev.2 - is a relatively small architecture with only 4137 trainables parameters."
"rev.3 - is upgraded model. This architechture is inspired by VGG model, but it is strongly reduced and tuned."
"EfficientNetB0 - is Efficient Net B0 model from Keras. It is trained on real truck data with extended background samles"
st.subheader("The ML models are trained for different data")
"Trial data - dataset mix of cuttings recorded in the office and diverse background sounds."
"Realicstic Truck Data - is the dataset recorded using REAL truck."
st.subheader("The Realicstic Truck Dataset includes two microphones")
"Condenser - is a regular microphone listening acoustic waves if the air inside the truck space."
"Piezo - is a picrophone mounted on the surface of the canvas, therefore more sensitive to waves inside a canvas and less sensitive to air waves."
' '
' '

selected_ml_model = st.selectbox(label="Select ML model architecture and Data on which it is trained here:", 
    options=[
    "ML model Rev.2 >> Trial Data",

    "ML model EfficientNetB0 97acc>> Realicstic Truck Data >> Piezo Mic",
    "ML model EfficientNetB0 upd1: 95acc >> Realicstic Truck Data >> Piezo Mic",

    "ML model Rev.2 >> Realicstic Truck Data >> Piezo Mic",
    "ML model Rev.2 >> Realicstic Truck Data >> Condenser Mic",

    "ML model Rev.3 >> Realicstic Truck Data >> Piezo Mic",
    "ML model Rev.3 >> Realicstic Truck Data >> Condenser Mic",
    ])


# ----------------------------------
# ==== Load ML model and see it ====
# ----------------------------------


## Clean memory, remove models
##--------------------------------
tf.keras.backend.clear_session() # clean memory, remove models


## Load the model
##--------------------------------
if selected_ml_model == "ML model Rev.2 >> Trial Data":
    reloaded_model = tf.keras.models.load_model("./tf_models/modelTN2/modelTN2")

if selected_ml_model == "ML model EfficientNetB0 97acc>> Realicstic Truck Data >> Piezo Mic":
    reloaded_model = tf.keras.models.load_model("./tf_models/efficientnetB0")

if selected_ml_model == "ML model EfficientNetB0 upd1: 95acc >> Realicstic Truck Data >> Piezo Mic":
    reloaded_model = tf.keras.models.load_model("./tf_models/efficientnet_95acc")

if selected_ml_model == "ML model Rev.2 >> Realicstic Truck Data >> Piezo Mic":
    reloaded_model = tf.keras.models.load_model("./tf_models/modelTN_dsr_data_it4_475samples_16bit_piezo")
if selected_ml_model == "ML model Rev.2 >> Realicstic Truck Data >> Condenser Mic":
    reloaded_model = tf.keras.models.load_model("./tf_models/modelTN_dsr_data_it4_475samples_16bit_condenser")

if selected_ml_model == "ML model Rev.3 >> Realicstic Truck Data >> Piezo Mic":
    reloaded_model = tf.keras.models.load_model("./tf_models/modelTN__vgg_s8-64_l2233_s09e5__data_it4_475samples_16bit__piezo")
if selected_ml_model == "ML model Rev.3 >> Realicstic Truck Data >> Condenser Mic":
    reloaded_model = tf.keras.models.load_model("./tf_models/modelTN__vgg_s8-64_l2233_s195e4__data_it4_475samples_16bit__condenser")






## Check model architecture
##--------------------------------
model_summary_stringlist = []
reloaded_model.summary(print_fn=lambda x: model_summary_stringlist.append(x))
short_model_summary = "\n".join(model_summary_stringlist)
# print(short_model_summary)
st.markdown(" ##### _ML model architecture_")
st.code(body=short_model_summary, language="Python")



# -----------------------------------
# ==== Predict with loaded Model ====
# -----------------------------------
# multiple if-statements are needed to adjust inputs for models

if selected_ml_model == "ML model EfficientNetB0 97acc>> Realicstic Truck Data >> Piezo Mic":
    spectrogram_shape_to_analyze_ENetB0 = (64, 64)
    spectrogram_arr_resized_ENetB0 = tf.image.resize(spectrogram_arr, spectrogram_shape_to_analyze_ENetB0)
    y_pred_ENetB0 = reloaded_model.predict(np.expand_dims(spectrogram_arr_resized_ENetB0, 0))
    audio_data_predicted_label = 1 - np.round(y_pred_ENetB0[0,0], decimals=2) #1-val bcoz model trained as 0=event, 1=bkg

if  selected_ml_model == "ML model EfficientNetB0 upd1: 95acc >> Realicstic Truck Data >> Piezo Mic":
    spectrogram_shape_to_analyze_ENetB0 = (128, 64)
    spectrogram_arr_resized_ENetB0 = tf.image.resize(spectrogram_arr, spectrogram_shape_to_analyze_ENetB0)
    y_pred_ENetB0 = reloaded_model.predict(np.expand_dims(spectrogram_arr_resized_ENetB0, 0))
    audio_data_predicted_label = 1 - np.round(y_pred_ENetB0[0,0], decimals=2) #1-val bcoz model trained as 0=event, 1=bkg

if selected_ml_model == 'ML model Rev.2 >> Trial Data':
    y_pred_1 = reloaded_model.predict(np.expand_dims(spectrogram_arr_resized, 0))
    audio_data_predicted_label = 1 - np.round(y_pred_1[0,0], decimals=2) #1-val bcoz model trained as 0=event, 1=bkg

if selected_ml_model == "ML model Rev.2 >> Realicstic Truck Data >> Piezo Mic" or \
        selected_ml_model == "ML model Rev.2 >> Realicstic Truck Data >> Condenser Mic" or \
        selected_ml_model == "ML model Rev.3 >> Realicstic Truck Data >> Piezo Mic" or \
        selected_ml_model == "ML model Rev.3 >> Realicstic Truck Data >> Condenser Mic" :
    y_pred_2 = reloaded_model.predict(np.expand_dims(spectrogram_arr_resized, 0))
    audio_data_predicted_label = np.round(y_pred_2[0,0], decimals=2)



 




st.subheader("Prediction:")

pred_index = np.round(audio_data_predicted_label, decimals=0).astype(int)
results_options = ['No cutting sound detected.',
                    'Canvas cutting is DETECTED.']

st.markdown(f"### _{results_options[pred_index]}_")
# st.write(f"Result: {results_options[pred_index]}")



## Output of more tech data
##--------------------------------
st.markdown(f"#### Model Prediction Output: {audio_data_predicted_label}")
print(f"Model Prediction Output: {audio_data_predicted_label}")

st.markdown(f"#### Model Prediction Output Index: {pred_index}")
print(f"Model Prediction Output Index: {pred_index}")

st.write(uploaded_audio_file_4debug)
print(uploaded_audio_file_4debug)
