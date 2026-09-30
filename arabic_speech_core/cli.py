from __future__ import annotations
import argparse, json, os
from .gates import text_gate, provider_preflight, tts_smoke_test

def main():
    p=argparse.ArgumentParser(); p.add_argument("command", choices=["preflight"]); p.add_argument("--text", required=True); p.add_argument("--lexicon"); p.add_argument("--engine", default=os.getenv("TTS_ENGINE","edge")); p.add_argument("--locale", default=os.getenv("LOCALE","ar-SA")); p.add_argument("--voice", default=os.getenv("EDGE_TTS_VOICE","ar-SA-HamedNeural")); p.add_argument("--min-words", type=int, default=1); p.add_argument("--max-words", type=int, default=2500); p.add_argument("--no-smoke", action="store_true")
    a=p.parse_args(); result=text_gate(a.text,min_words=a.min_words,max_words=a.max_words,lexicon_path=a.lexicon); result["provider"]=provider_preflight(engine=a.engine,locale=a.locale,voice=a.voice)
    if not a.no_smoke: result["smoke"]=tts_smoke_test("هذه عينة صوت عربية قصيرة لاختبار المزود قبل التشغيل.",engine=a.engine,voice=a.voice)
    print(json.dumps(result, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
