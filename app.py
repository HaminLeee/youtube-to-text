import torch
# import torchaudio
from glob import glob
from flask import Flask,render_template,request
import pafy
import os
import re


YOUTUBE_REGEX="^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$"
app = Flask(__name__, template_folder="templates")

@app.route("/", methods=[ "GET", "POST" ])
def hello_world():
    # print(speech2text())
    # transcript = speech2text("./audios/out/out.wav")
    transcript, youtube_response, youtube_link, loading = "", "", "", ""
    if request.method == "POST":
        youtube_link = request.form.get("youtube_link")
        youtube_regex = YOUTUBE_REGEX
        loading = "loading..."
        video = youtube_parser(youtube_link)
        youtube_response = 'Title: {}, UserName: {}, Views: {}, Likes: {}'.format(video.title, video.username, video.viewcount, video.likes) if re.match(youtube_regex, youtube_link) != None else 'Invalid Link'
        transcript = speech2text("./audios/out/out.wav")

        loading = ""
        return render_template("index.html", transcript=transcript, youtube_link=youtube_link, thumbnail=video.thumb, youtube_response=youtube_response, loading=loading)

    if request.method == "GET":
        transcript = speech2text("./audios/out/out.wav")
        return render_template("index.html", transcript=transcript, youtube_link=youtube_link, youtube_response=youtube_response)

def youtube_parser(url):    
    os.system("rm -f ./audios/in/in.m4a")
    # instant created
    video = pafy.new(url)
    
    audiostreams = video.audiostreams
    print(audiostreams)
    
    bestaudio = video.getbestaudio()
    bestaudio.download("./audios/in/in.m4a")

    os.system("ffmpeg -i ./audios/in/in.m4a ./audios/out/out.wav -y")
    os.system("rm -f ./audios/in/in.m4a")

    return video


def speech2text(input_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu also works, but our models are fast enough for CPU

    model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                        model='silero_stt',
                                        language='en', # also available 'de', 'es'
                                        device=device)
    (read_batch, split_into_batches,
    read_audio, prepare_model_input) = utils  # see function signature for details

    test_files = glob(input_file)
    batches = split_into_batches(test_files, batch_size=10)
    input = prepare_model_input(read_batch(batches[0]),
                                device=device)
    print("===== Speech2Text Started ... =====")
    output = model(input)
    res = ""
    for example in output:
        res += decoder(example.cpu())
    print("\n\noutput: " + res + "\n\n")
    print("Aborting.. \n\n===== Speech2Text Done =====")

    return res


if __name__ == "__main__":
    app.run()