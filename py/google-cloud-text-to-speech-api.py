# --------------------------------------------------------------------------
#  To install the library:
#       $ pip install --upgrade google-cloud-texttospeech
# --------------------------------------------------------------------------

# ---------------------
# 0) Load credentials
# ---------------------
import os
json_file = ''
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_file

# ---------------------
# 1) Initialize
# ---------------------

# Here's a simple text
sample_text = '''The job of capital markets is to process information
so that savings flow to the best projects and firms.
That makes high finance sound simple; in reality it is dynamic and intoxicating.
It reflects a changing world.'''

# Import the library
from google.cloud import texttospeech

# Instantiate a client
client = texttospeech.TextToSpeechClient()
# and the SynthesisInput object
input_text = texttospeech.types.SynthesisInput(text=sample_text)

# ---------------------
# 2) Set the voice
# ---------------------

# List all the available voices
print(client.list_voices())

# Define the voice, here we choose a Male WaveNet voice
voice = texttospeech.types.VoiceSelectionParams(
      language_code = 'fr-FR',
      name          = 'fr-FR-Standard-A',
      # language_code = 'ru-RU',
      # name          = 'ru-RU-Standard-B',
      ssml_gender   = texttospeech.enums.SsmlVoiceGender.FEMALE)

# -------------------------------
# 3) Define ouput encoding
# -------------------------------

# Set the encoding of the resulting audio file to mp3
audio_config = texttospeech.types.AudioConfig(
    audio_encoding=texttospeech.enums.AudioEncoding.MP3
)

# -----------------------------------------------
# 4) -- Query the API and play generated audio
# -----------------------------------------------

response = client.synthesize_speech( \
    input_text, \
    voice, \
    audio_config)

# Write the binary response to an mp3 file
with open('sample.mp3', 'wb') as out:
  out.write(response.audio_content)

# and play the gererated audio file
import os
print(sample_text)
os.system('play sample.mp3')





# # df
# # import pandas and bigquery libraries
# import pandas as pd
# import google.datalab.bigquery as bq
#
# # define SQL query
# sql = "SELECT * FROM `fh-bigquery.reddit_posts.2017_11` WHERE subreddit = 'Jazz'"
#
# # Query BigQuery and load results into a pandas dataframe
# df = bq.Query(sql).execute().result().to_dataframe()
# cols = ['id', 'title', 'selftext', 'author', 'domain', 'url', 'num_comments',
#        'score',  'thumbnail', 'is_self',  'permalink', 'created_utc']
# df = df[cols]
# cond = (df.selftext != '') & (df.selftext != '[deleted]')
#
# df[cond].selftext.head()
#
#
# # list of wavenet voices in english
# voices = []
# for v in client.list_voices().voices:
#     if ('en-US' in v.language_codes ) & ('Wavenet' in v.name):
#         voices.append(v)
#
# for i, d in df[cond][3:6].iterrows():
#     print('=='*50)
#     voice = random.choice(voices)
#     tts_voice = texttospeech.types.VoiceSelectionParams(
#           language_code='en-US',
#           name = voice.name,
#           ssml_gender=voice.ssml_gender)
#
#     print('--'*20)
#     print(voice)
#     print('--'*20)
#     print(d.selftext)
#     print('--'*20)
#     mp3_filename  = "{}.mp3".format(d.id)
#     input_text = texttospeech.types.SynthesisInput(text=d.selftext)
#     response = client.synthesize_speech(input_text, tts_voice, audio_config)
#     with open(mp3_filename, 'wb') as out:
#         out.write(response.audio_content)
#     print('-- playing {}'.format(mp3_filename))
#     os.system('play {}'.format(mp3_filename))
