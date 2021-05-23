from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

TEXT = "A list of open speech corpora for Speech Technology research and development."
# use_cuda = True

DEBUG = True
use_cuda = False


def synth(text="hello world."):

    TTS_CONFIG = "voiceModel/tts_models--en--ljspeech--tacotron2-DDC/config.json"
    TTS_MODEL = "voiceModel/tts_models--en--ljspeech--tacotron2-DDC/model_file.pth.tar"

    VOCODER_MODEL = "voiceModel/vocoder_model--en--ljspeech-hifigan_v2/model_file.pth.tar"
    VOCODER_CONFIG = "voiceModel/vocoder_model--en--ljspeech-hifigan_v2/config.json"

    model_path = TTS_MODEL
    config_path = TTS_CONFIG
    vocoder_path = VOCODER_MODEL
    vocoder_config_path = VOCODER_CONFIG

    synthesizer = Synthesizer(model_path, config_path, vocoder_path, vocoder_config_path, use_cuda)

    print(" > Text: {}".format(text))

    wav = synthesizer.tts(text)

    if DEBUG:
        out_path = "out/1.wav"
        synthesizer.save_wav(wav, out_path)

    return wav


if __name__ == "__main__":
    synth()
