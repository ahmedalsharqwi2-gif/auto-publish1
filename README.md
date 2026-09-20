# Auto Publish Pipeline

أتمتة كاملة لنشر المحتوى على وسائل التواصل الاجتماعي عبر Buffer API.

## 🌟 المميزات

- ✅ نشر تلقائي على Facebook, Twitter, LinkedIn عبر Buffer
- ✅ تكامل مع GitHub Actions للجدولة
- ✅ تتبع المواضيع المنشورة في `topic_history.json`
- ✅ دعم الصور والنصوص
- ✅ إعدادات بسيطة في ملف `config.py`

## 📁 هيكل المشروع

```
auto-publish1/
├── .github/
│   └── workflows/
│       └── auto_publish.yml    # GitHub Actions workflow
├── assets/                      # الصور والملفات медиа
├── config.py                    # الإعدادات و API Keys
├── main.py                      # الكود الرئيسي
├── requirements.txt             # المكتبات المطلوبة
└── topic_history.json          # سجل المواضيع المنشورة
```

## 🚀 التثبيت

### 1. استنساخ المشروع

```bash
git clone https://github.com/ahmedalsharqwi2-gif/auto-publish1.git
cd auto-publish1
```

### 2. تثبيت المكتبات

```bash
pip install -r requirements.txt
```

### 3. إعداد المتغيرات البيئية

أنشئ ملف `.env` أو اضبط المتغيرات التالية:

```bash
BUFFER_ACCESS_TOKEN=your_buffer_token_here
```

### 4. تعديل الإعدادات

افتح `config.py` وعدّل:

```python
BUFFER_ACCESS_TOKEN = "your_token_here"
PROFILES = ["profile_id_1", "profile_id_2"]  # Optional
TOPICS_FILE = "topic_history.json"
```

## 📖 الاستخدام

### تشغيل يدوي

```bash
python main.py
```

### جدولة مع GitHub Actions

الكود مضبوط ليعمل تلقائياً عبر GitHub Actions. لتعديل الجدول:

1. افتح `.github/workflows/auto_publish.yml`
2. عدّل `cron` expression:

```yaml
schedule:
  - cron: '0 */6 * * *'  # كل 6 ساعات
```

## 🔧 Troubleshooting

### خطأ: `ModuleNotFoundError`

```bash
pip install -r requirements.txt --upgrade
```

### خطأ: `Buffer API authentication failed`

- تأكد من أن `BUFFER_ACCESS_TOKEN` صحيح
- تحقق من صلاحيات الـ Token في Buffer Dashboard

### خطأ: `No topics available`

- تأكد أن `topic_history.json` يحتوي على مواضيع
- أو أضف مواضيع جديدة يدوياً

## 📊 مثال على topic_history.json

```json
{
  "posted_topics": [
    {
      "topic": "أفضل 5 نصائح للإنتاجية",
      "posted_at": "2026-09-20T10:00:00Z",
      "platforms": ["facebook", "twitter"]
    }
  ],
  "pending_topics": [
    "كيف تبدأ في البرمجة؟",
    "أدوات الذكاء الاصطناعي للمبتدئين"
  ]
}
```

## 🔐 الأمان

- ⚠️ **لا تشارك** `BUFFER_ACCESS_TOKEN` أبداً
- ✅ استخدم GitHub Secrets للـ CI/CD
- ✅ أضف `.env` إلى `.gitignore`

## 📝 الترخيص

MIT License — حر في الاستخدام والتعديل.

## 🤝 المساهمة

1. Fork المشروع
2. أنشئ فرع جديد (`git checkout -b feature/NewFeature`)
3. Commit التغييرات (`git commit -m 'Add NewFeature'`)
4. Push للفرع (`git push origin feature/NewFeature`)
5. افتح Pull Request

## 📧 للتواصل

- GitHub: [@ahmedalsharqwi2-gif](https://github.com/ahmedalsharqwi2-gif)

---

**مبني بحب ❤️ للأتمتة والإنتاجية**
