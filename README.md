# Auto Publish Reels

خط أنابيب آلي لإنشاء مقاطع Reels عربية قصيرة، مع التحقق من الحقائق، تحويل النص إلى صوت، وإرسال الناتج إلى قنوات النشر.

## هيكل المشروع

- `main.py` — نقطة تشغيل خط الأنابيب.
- `llm_gemini.py` — عميل Gemini المستخدم لتوليد النصوص.
- `fact_check.py` — التحقق من الحقائق والمصادر.
- `tts_quality.py` — توليد الصوت وفحص جودته.
- `make_reference_voice.py` — إنشاء عينة صوت مرجعية عند الحاجة.
- `config/` — مصادر الحقائق وبنك الموضوعات.
- `assets/voices/` — ملفات وأوصاف أصوات SILMA.
- `tools/` — أدوات مساعدة مثل بناء بنك الموضوعات.
- `tests/` — اختبارات المشروع.
- `.github/workflows/` — مهام GitHub Actions للنشر وإنشاء الصوت المرجعي.

## التشغيل

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py'
```

يتطلب التشغيل الكامل تثبيت `ffmpeg` وضبط مفاتيح الخدمات المطلوبة كمتغيرات بيئية أو أسرار GitHub Actions. لا تُحفظ الملفات الناتجة أو الأسرار داخل المستودع.
