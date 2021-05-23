import json

import synth
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

app = FastAPI()


j = """{
    "text": "hello world",
    "textToSpectrogramModel": "Tacotron2-CassieLeeMorris",
    "vocoder": "hifi_gan",
    "sampleRate": 22050,
    "responseFormat": ""
}"""


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        text = None
        try:
            jsonData = json.loads(data)
            text = jsonData["text"]
            wav = synth.synth(text)

        except KeyError:
            pass
            # TODO:handle error

        except json.decoder.JSONDecodeError:
            pass
            # TODO:handle error

        await websocket.send_text(f"Message text was: {text}")
        print(type(wav))  # list type
        # await websocket.send_bytes(wav) # doesn't work yet
