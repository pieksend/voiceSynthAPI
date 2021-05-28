import json

import synth
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()

import numpy as np

import io
import base64
import asyncio

import scipy
from scipy.io.wavfile import write

from hypercorn.config import Config
from hypercorn.asyncio import serve

config = Config()
config.bind = ["localhost:8080"]  # As an example configuration setting


from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


import torch
import os.path
import os

class GlowTTSSynthesis:
    def __init__(self) -> None:
        cwd = os.getcwd()
        model_path = str(os.path.join(cwd,"PretrainedModels","pretrained.pth"))
        config_path = str(os.path.join(cwd,"PretrainedModels","glowtts-config.json"))
        vocoder_path = str(os.path.join(cwd,"PretrainedModels","hifigan-458080.pth.tar"))
        vocoder_config_path = str(os.path.join(cwd,"PretrainedModels","hifigan-config.json"))

        self.samplingRate = 22050

        if torch.cuda.is_available():
            self.useCuda = True
        else:
            self.useCuda = False

        self.synthesizer = Synthesizer(tts_checkpoint= model_path,tts_config_path=config_path,vocoder_checkpoint = vocoder_path,vocoder_config=vocoder_config_path,use_cuda=self.useCuda)

    def textToSpeechInference(self,text:str):
        wav = self.synthesizer.tts(text)
        wavArray = np.array(wav)
        bytesArray = GlowTTSSynthesis.toBytesFromAudio(wavArray,samplingRate=self.samplingRate)
        return bytesArray

    @staticmethod
    def toBytesFromAudio(audio:np.array,samplingRate:int):
        bytes_wav = bytes()
        byte_io = io.BytesIO(bytes_wav)
        write(byte_io, samplingRate, audio)
        result_bytes = byte_io.read()
        return result_bytes

# Demo app.
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Text to Speech Inference</title>
    </head>
    <body>
        <h1>Text to Speech Inference</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Text to Speech</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>

            var ws = new WebSocket("ws://localhost:8080/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                jsonObj = JSON.parse(event.data)
                var text = jsonObj["text"]
                var audio = jsonObj["audio"]

                // Create the text to be spoken
                var content = document.createTextNode(text)
                message.appendChild(content)
                messages.appendChild(message)
                // decode array and then play
                bytes = _base64ToArrayBuffer(audio)
                play(bytes)
            };

            function sendMessage(event) {
                var input = document.getElementById("messageText")
                var data = { text:input.value, sampleRate:22050, textToSpectrogramModel:"Tacotron2-CassieLeeMorris",
                             vocoder: "hifi-gan", responseFormat:"wav" }
                ws.send(JSON.stringify(data))

                input.value = ''
                event.preventDefault()
            }

            function _base64ToArrayBuffer(base64) {
                var binary_string = window.atob(base64);
                var len = binary_string.length;
                var bytes = new Uint8Array(len);
                for (var i = 0; i < len; i++) {
                    bytes[i] = binary_string.charCodeAt(i);
                }
                return bytes.buffer;
            }

            function play(data) {
                const context = new AudioContext();
                context.sampleRate = 22050 
                context.decodeAudioData(data,
                function (audioData) {
                    const buffer = audioData;
                     const source = context.createBufferSource();
                    source.buffer = buffer;
                    source.connect(context.destination);
                    source.start();
                }, 
                function (error) {
                    console.log(error);
                });
               
             }
        </script>
    </body>
</html>
"""


tts = None

@app.on_event("startup")
async def startup_event():
    global tts 
    tts = GlowTTSSynthesis()


@app.get("/")
async def get():
    return HTMLResponse(html)

htmlText = html

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        text = None

        try:
            jsonData = json.loads(data)
            text = jsonData["text"]
            sampleRate = jsonData["sampleRate"]

            #wav = synth.synth(text)

        except KeyError:
            pass
            # TODO:handle error

        except json.decoder.JSONDecodeError:
            pass
            # TODO:handle error

        # TODO call inference from here 

    
        wavBytes = tts.textToSpeechInference(text=text) # audio 16 bits type # fill in here. and pass in sample rate.

        base64_bytes = base64.b64encode(wavBytes)
        base64_bytes = base64_bytes.decode('ascii') # encode it as a string Of Words

        jsonData = {}
        jsonData["audio"] = base64_bytes # will be decoded on the other side, use localhost:8080 / to load the page and it will decode there when performing inference
        jsonData["text"] = data
        jsonString = json.dumps(jsonData)
        await websocket.send_text(jsonString)


def saveToFile(self,file:str, samplingRate, audio):
    write(file, samplingRate, audio)

if __name__ == "__main__":
    asyncio.run(serve(app, config))
