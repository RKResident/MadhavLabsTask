import json
import glob
import os

['album_artists', 'artists', 'forms', 'layas', 'length', 'mbid', 'raags', 'release', 'taals', 'title', 'works']
['album_artists', 'artists', 'concert', 'form', 'length', 'mbid', 'raaga', 'taala', 'title', 'work']

thaats = {
  "Miyām̐ malhār": "kafi",
  "Bhūp": "bilawal",
  "Suhā": "kafi",
  "Khokar": "khamaj",
  "Śrī": "poorvi",
  "Bhairav": "bhairav",
  "Bibhās": "bhairav",
  "Bhimapalās": "kafi",
  "Mālkauns": "bhairavi",
  "Trivēṇī gauri": "poorvi",
  "Bihāg": "bilawal",
  "Rāgēśrī": "khamaj",
  "Ābhōgī": "kafi",
  "Mārvā": "marwa",
  "Lalit": "poorvi",
  "Bhairavi": "bhairavi",
  "Jait kalyāṇ": "kalyan",
  "Multāni": "todi",
  "Bilāsakhānī tōḍī": "todi",
  "Gaurī": "poorvi",
  "Rāmdāsī malhār": "bhairav",
  "Gauḍ malhār": "khamaj",
  "Basantī kēdār": "kalyan",
  "Ahira bhairav": "bhairav",
  "Jōg": "khamaj",
  "Kēdār": "kalyan",
  "Madhukauns": "bhairavi",
  "Hindōl pañcam": "kalyan",
  "Kīravāṇi": "kalyan",
  "Bairāgi": "bhairav",
  "Śuddh kalyāṇ": "kalyan",
  "Dhanī": "kafi",
  "Yaman kalyāṇ": "kalyan",
  "Kalāvati": "khamaj",
  "Naṭ kāmōd": "khamaj",
  "Dēś": "khamaj",
  "Lagan gāndhār": "marwa",
  "Sarasvati": "kalyan",
  "Puriyā dhanaśrī": "marwa",
  "Tōḍī": "todi",
  "Miśra kaliṅgaḍā": "bhairavi",
  "Khaṭa": "bhairavi",
  "Naṭ bhairav": "bhairav",
  "Lalit pañcam": "poorvi",
  "Kōmal riṣabh asāvēri": "bhairavi",
  "Bhāṭiyār": "marwa",
  "Gavti": "kafi",
  "Hamīr": "bilawal",
  "Mārūbihāg": "bilawal",
  "Kalyāṇ": "kalyan",
  "Jōgiyā": "bhairav",
  "Miśra pīlū": "khamaj",
  "Ḍāgori": "asavari",
  "Khamāj": "khamaj",
  "Mājh khamāj": "khamaj",
  "Candrakauns": "bhairavi",
  "Sōhinī": "marwa",
  "Bāgēśrī": "kafi",
  "Śuddh sāraṅg": "bilawal",
  "Bahār": "kafi",
  "Mēgh": "kafi"
}
d = {}
for dir in os.listdir('/home/sabyasachi19/PycharmProjects/MusicML/data/hindustani'):
    try:
        fname = glob.glob(os.path.join('/home/sabyasachi19/PycharmProjects/MusicML/data/hindustani', dir, '*.json'))[0]
    except:
        print("skipping", dir)
        break
    # print(fname)
    with open(fname, 'r') as f:
        data = json.load(f)
        try:
            if data["raaga"][0]["thaat"] in d:
                d[data["raaga"][0]["thaat"]] += 1
            else:
                d[data["raaga"][0]["thaat"]] = 1
        except:
            pass

print(len(d))
# print(rags)
# print(sum([rags[elem] for elem in rags]))
# print(list(rags.keys()))