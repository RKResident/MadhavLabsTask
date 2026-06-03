Generated embeddings are all in output folder. Run script generate_embeddings.py to generate future
embeddings.

args to be passed with generate_embeddings:
* --artist to filter by artist
* --raaga
* --taala
* --concert (concert name)
* --base_dir (data/carnatic or data/hindustani as need be)
* --chunk_sec (duration of each chunk)
* --layer embeddings to be taken from which mert layer, integer from -1 to 12

for visualizaion, upload to https://projector.tensorflow.org"

Use script analyze_metadata.py to give stats about each directory. Pass arg --base_dir

Training procedure

Tracks are sliced into 30s long windows. These are our actual training+testing sample points.
This length was chosen as it would be long enough to run over a full raaga phrase.

Model for classification is inspired from https://arxiv.org/pdf/1706.02921. It has first 6 layer of
MERT as fixed weights, following by 5 2d conv layers with window size 5. Earlier i'd tried 1d conv
with first 6 MERT layers, but it resulted in extreme overfitting within first epoch itself.

My experimentation

The biggest issue with this task is the lack of data for each raaga. In the hindustani dataset,
most songs have raags, but each unique raag has just 1-3 songs, with most having just 1.

Additionally, we can't just include windows from same channel in train and test, as it would result
in model learning features that aren't actually raag features, but those specific to the particular song.

This was even more evidences by my embedding visualizations - i experimented with multiple intermediate
embeddings, but the embeddings always tend to cluster by artist/song first. If filtered by artist, 
they'd cluster by concert.

Back to lack of data, since each raag has just 1-2 tracks, I believe it's better to classify into 
groups of raags, called thaats. While there are 84 standard hindustani raagas, there's only 10 thaats.
With a dataset of 106 songs, it's more practical to train on thaats.

Even with multiple architecture changes, the model overfits early and eval accuracy stays extremely low.