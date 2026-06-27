"""Language registry for LanguageTranslator.

Maps user-friendly names / aliases -> (whisper_lang_code, display_name, initial_prompt).
initial_prompt in the target language primes the decoder vocabulary for better accuracy.
"""

LANGUAGES = {
    # English
    "english": ("en", "English",
        "These are notes from a meeting or personal dictation."),
    "en": ("en", "English",
        "These are notes from a meeting or personal dictation."),

    # German
    "german": ("de", "Deutsch",
        "Dies sind Notizen aus einem Meeting oder einer persönlichen Diktat. "
        "Protokoll, Besprechung, Aufgabe, Termin, Zusammenfassung."),
    "deutsch": ("de", "Deutsch",
        "Dies sind Notizen aus einem Meeting oder einer persönlichen Diktat. "
        "Protokoll, Besprechung, Aufgabe, Termin, Zusammenfassung."),
    "de": ("de", "Deutsch",
        "Dies sind Notizen aus einem Meeting oder einer persönlichen Diktat."),

    # Finnish
    "finnish": ("fi", "Suomi",
        "Nämä ovat muistiinpanoja kokouksesta tai henkilökohtaisesta sanelusta. "
        "Kokous, tehtävä, muistio, yhteenveto, aikataulu, projekti."),
    "suomi": ("fi", "Suomi",
        "Nämä ovat muistiinpanoja kokouksesta tai henkilökohtaisesta sanelusta."),
    "fi": ("fi", "Suomi",
        "Nämä ovat muistiinpanoja kokouksesta tai henkilökohtaisesta sanelusta."),

    # French
    "french": ("fr", "Français",
        "Ce sont des notes de réunion ou de dictée personnelle. "
        "Réunion, tâche, résumé, agenda, projet, compte-rendu."),
    "français": ("fr", "Français",
        "Ce sont des notes de réunion ou de dictée personnelle."),
    "fr": ("fr", "Français",
        "Ce sont des notes de réunion ou de dictée personnelle."),

    # Spanish
    "spanish": ("es", "Español",
        "Estas son notas de una reunión o dictado personal. "
        "Reunión, tarea, resumen, proyecto, agenda, nota."),
    "español": ("es", "Español",
        "Estas son notas de una reunión o dictado personal."),
    "es": ("es", "Español",
        "Estas son notas de una reunión o dictado personal."),

    # Hindi
    "hindi": ("hi", "हिन्दी",
        "ये एक बैठक या व्यक्तिगत श्रुतलेख के नोट्स हैं।"),
    "hi": ("hi", "हिन्दी",
        "ये एक बैठक या व्यक्तिगत श्रुतलेख के नोट्स हैं।"),

    # Japanese
    "japanese": ("ja", "日本語",
        "これは会議または個人的な口述のメモです。会議、タスク、要約、スケジュール。"),
    "ja": ("ja", "日本語",
        "これは会議または個人的な口述のメモです。"),

    # Chinese (Mandarin)
    "chinese": ("zh", "中文",
        "这是会议或个人口述的笔记。会议、任务、摘要、日程。"),
    "zh": ("zh", "中文",
        "这是会议或个人口述的笔记。"),

    # Auto-detect
    "auto": (None, "Auto-detect", ""),
}


def resolve_language(lang: str) -> tuple[str | None, str, str]:
    """Resolve a language name/alias to (whisper_lang_code, display_name, initial_prompt).

    Falls back to English for unknown names.
    """
    key = lang.lower().strip()
    if key not in LANGUAGES:
        return LANGUAGES["english"]
    return LANGUAGES[key]
