'''

see https://cloud.google.com/bigquery/docs/reference/libraries#client-libraries-install-python

# Reddit Jazz
SELECT * FROM [fh-bigquery:reddit_posts.2017_11] WHERE subreddit = 'Jazz'

export N posts and comments into BQ

# Vision
- extract urls or images from posts and detect:

* label detection: core
* Face detection: is that a face, where in the image + emotion
* OCR: text in an image, lang
* Explicit content detection: user generated content
* Landmark detection: common landmark, lat long
* Logo Detection: id logos
* web annotations

'''
import os
import pandas as pd
# import google.datalab.bigquery as bq
from google.cloud import bigquery as bq

json_file = '/Users/alexis/amcp/credentials/Datascience-94698039443a.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_file
client = bq.Client()

# ------------------------------------------------------
#  Load the data into a dataframe
# ------------------------------------------------------

# Original BQ query
sql = 'SELECT * FROM "fh-bigquery:reddit_posts.2017_11" WHERE subreddit = "Jazz"'
# Query to use in python
sql = "SELECT * FROM `fh-bigquery.reddit_posts.2017_11` WHERE subreddit = 'Jazz' limit 200 "

df = client.query(sql).to_dataframe()


df.head(10)
# Kepp most important columns
cols = ['id', 'title', 'selftext', 'author', 'url', 'permalink', 'created_utc']
df = df[cols]

# Find all images from the url field
# Get all the urls with images

cond = df.url.str.contains('png') | df.url.str.contains('jpg')
print("{} images".format(df[cond].shape[0]))
df[cond].shape
df[cond].url.head(15)

'''we have 57 images'''

# enable Vision API
# https://console.cloud.google.com/apis/library/vision.googleapis.com/?project=datascience-194017&folder&organizationId

# see https://google-cloud-python.readthedocs.io/en/latest/vision/gapic/v1/types.html#google.cloud.vision_v1.types.AnnotateImageResponse

# Import the Google Cloud client library
from google.cloud import vision

# Instantiate a client
client = vision.ImageAnnotatorClient()

# define the request parameters
url = 'http://digitalspyuk.cdnds.net/13/06/980x653/gallery_movies-anchorman-12.jpg'

url = 'https://i.redd.it/vem49j4u72yz.jpg'
url = 'https://i.redd.it/koqmyity3vxz.png'
request = {
    'image':  {
        'source':  {
            'image_uri': url
            }
        }
    }

res = client.annotate_image(request)

print(res.label_annotations)
print(res.text_annotations)
# ...

'''
Let's store the label_annotations in a new columns, separated by a;
let's build a new dict which we'll add to our original dataframe in a second step
working with dict is faster than adding data to dataframe when the dataframe is big
'''
from tqdm import tqdm


def extract_text(text_annotations):
    '''
    returns the text extracted by the OCR module if any
    '''
    if len(text_annotations) > 0:
        return text_annotations[0].description
    else:
        return ''

def extract_labels(label_annotations, threshold):
    '''
    returns the list of label descriptions found with a score > threshold
    '''
    labels = [label.description for label in res.label_annotations if label.score > threshold]
    return labels

def extract_web(web_entities, threshold):
    '''
    returns the list of web entities found with a score > threshold
    '''
    labels = [web.description for web in  web_entities if web.score > threshold]
    return labels

def json_query(url):
    return {
        'image':  {
            'source':  {
                'image_uri': url
                }
            },
        'features': [
            {'type': vision.enums.Feature.Type.LABEL_DETECTION},
            {'type': vision.enums.Feature.Type.TEXT_DETECTION},
            {'type': vision.enums.Feature.Type.WEB_DETECTION},
            ],
        }


# initialize the list
dlabels = []
# Loop over each dataframe row
for i,d in df[cond].iterrows():

    request = json_query(d.url)
    res     = client.annotate_image(request)

    dlabels.append( {
        'id'    : d.id,
        'label' : extract_labels(res.label_annotations, 0.75 ),
        'text'  : extract_text(res.text_annotations),
        'web'   : extract_web(res.web_detection.web_entities, 0.5),
    })

# transform the list of dictionaries into a Dataframe and merge with original df
df = df.merge(pd.DataFrame(dlabels), on='id')

# interesting case: partition: https://i.redd.it/08b7ofzy0gzz.jpg with some text

url = 'https://i.redd.it/08b7ofzy0gzz.jpg'
print(res.text_annotations[0].description)

# -----------------------------------------------------------
'''
restrict to certain annotations types
?vision.enums.Feature.Type
Init signature: vision.enums.Feature.Type()
Docstring:
Type of image feature.

Attributes:
  FACE_DETECTION (int): Run face detection.
  LANDMARK_DETECTION (int): Run landmark detection.
  LOGO_DETECTION (int): Run logo detection.
  LABEL_DETECTION (int): Run label detection.
  TEXT_DETECTION (int): Run OCR.
  DOCUMENT_TEXT_DETECTION (int): Run dense text document OCR. Takes precedence when both DOCUMENT_TEXT_DETECTION and TEXT_DETECTION are present.
  SAFE_SEARCH_DETECTION (int): Run computer vision models to compute image safe-search properties.
  IMAGE_PROPERTIES (int): Compute a set of image properties, such as the image's dominant colors.
  CROP_HINTS (int): Run crop hints.
  WEB_DETECTION (int): Run web detection.
'''
request = {
    'image':  {
        'source':  {
            'image_uri': 'http://digitalspyuk.cdnds.net/13/06/980x653/gallery_movies-anchorman-12.jpg'
            }
        },
    'features': [
        { 'type': vision.enums.Feature.Type.FACE_DETECTION, 'maxResults': 2},
        { 'type': vision.enums.Feature.Type.LABEL_DETECTION, 'maxResults': 4},
        ],
    }

res = client.annotate_image(request)
print(res)

# method_list = [func for func in dir(res) if callable(getattr(res, func))]


# -----------------------------------------------------------
