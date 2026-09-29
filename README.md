# Auto Publish Reels

خطّ إنتاج مبسّط لإنشاء Reels عربية: يولّد موضوعًا ونصًا، يراجع النص والصوت، يركّب MP4 رأسيًا مع عناوين عربية، يرفع الفيديو إلى إصدار عام في GitHub، ثم يضيفه إلى قائمة Buffer لكل قناة محددة.

## مسار التشغيل

1. يولّد `scripts/generate_content.py` الموضوع والسرد عبر Gemini، ويطبّق مراجعة لغوية وفحص جودة النص.
2. يولّد `scripts/generate_voice.py` الصوت، ثم يفحص `scripts/quality_check.py` جودة النص والصوت.
3. ينشئ `scripts/assemble_reel.py` فيديو MP4 عموديًا بقياس 1080×1920، مع عنوان وترجمة عربية متزامنة تقديريًا مع الصوت.
4. يرفع `scripts/github_media.py` ملف MP4 إلى إصدار GitHub عام. يبقى ملف الإصدار متاحًا كي يتمكن Buffer من جلبه عند النشر.
5. ينشئ `scripts/publish_content.py` منشورًا لكل معرّف قناة عبر GraphQL API الرسمي لـBuffer، افتراضيًا بوضع `addToQueue`.

يجب أن يكون المستودع **عامًا** حتى يستطيع Buffer جلب ملف الفيديو. لا تحذف إصدارات الفيديو؛ فقد يكون المنشور ما زال في قائمة الانتظار.

## متطلبات التشغيل

يتطلب Python 3.11 أو أحدث، وFFmpeg مع الخط `Noto Sans Arabic`، ومفاتيح Gemini وBuffer، ومعرّفات قنوات Buffer الفعلية. يستخدم مسار الصوت الافتراضي Edge TTS بالصوت العربي المدعوم `ar-SA-HamedNeural`؛ ويمكن اختيار Google Cloud TTS عبر `TTS_ENGINE=google` إذا كانت بيانات اعتماده متاحة. يستخدم GitHub Actions رمز `GITHUB_TOKEN` بصلاحية `contents: write` لرفع ملف الفيديو إلى إصدار عام. لا يُستدعى Buffer أو يُرفع أي فيديو أثناء الاختبارات.

ثبّت المتطلبات الخفيفة وشغّل الاختبارات محليًا:

```bash
sudo apt-get install ffmpeg fonts-noto-core
python -m pip install -r requirements-runtime.txt
python -m unittest discover -s tests -p 'test_*.py' -v
```

للتشغيل المحلي، انسخ `.env.example` إلى `.env.local`، وأدخل الأسرار في بيئتك فقط، ثم شغّل `python main.py`. لا تضع أسرارًا في Git. في GitHub Actions، أضف `GEMINI_API_KEY` و`BUFFER_API_KEY` و`BUFFER_CHANNEL_IDS` كأسرار للمستودع. يجب أن يكون `BUFFER_CHANNEL_IDS` قائمة مفصولة بفواصل لمعرّفات القنوات، لا أسماء المنصات.

## إعدادات طول المقطع

يحدّد `MIN_WORDS` و`MAX_WORDS` حجم النص، ويحدّد `MAX_REEL_SECONDS` الحد الأعلى لمدة الصوت والفيديو. إعدادات workflow الحالية هي 90–165 كلمة وحد أقصى 90 ثانية. الفيديو يتبع مدة الصوت الفعلية ولا يُمدّد بصمت.

## سير العمل في GitHub

- `Auto publish Reels` يعمل وفق الجدول أو عند تشغيله يدويًا، ولا يُشغّل من أحداث Pull Request.
- `Arabic Guard and Unit Tests` يشغّل كامل الاختبارات عند كل Pull Request إلى `main`، ولا يحتوي على مفاتيح نشر أو خطوات نشر.
