# -*- coding: utf-8 -*-
"""
arабic Grammar Fixer - نظام إصلاح النحو العربي
========================================================
يصحح الأخطاء النحوية الشائعة ويحسن جودة النصوص قبل التحويل إلى كلام.
"""

from __future__ import annotations

import re
import logging
from typing import Dict, List, Tuple

log = logging.getLogger("pipeline")


class ArabicGrammarFixer:
    """محرك إصلاح النحو العربي المتقدم."""

    def __init__(self):
        """Initialize with common Arabic grammar rules."""
        self.rules = self._build_rules()

    def _build_rules(self) -> List[Tuple[re.Pattern, str, str]]:
        """Build comprehensive Arabic grammar rules.
        Each rule: (pattern, replacement, description)
        """
        return [
            # =========== عدم تطابق الضمائر والأفعال ===========
            (re.compile(r'\bهو\s+(?:قالت|ذهبت|كانت|فعلت|أكلت)\b'),
             lambda m: m.group(0).replace('قالت', 'قال').replace('ذهبت', 'ذهب')
                              .replace('كانت', 'كان').replace('فعلت', 'فعل')
                              .replace('أكلت', 'أكل'),
             'عدم تطابق الفعل المؤنث مع ضمير المذكر (هو)'),

            (re.compile(r'\bهي\s+(?:قال|ذهب|كان|فعل|أكل)\b'),
             lambda m: m.group(0).replace('قال', 'قالت').replace('ذهب', 'ذهبت')
                              .replace('كان', 'كانت').replace('فعل', 'فعلت')
                              .replace('أكل', 'أكلت'),
             'عدم تطابق الفعل المذكر مع ضمير المؤنث (هي)'),

            (re.compile(r'\bهم\s+(?:قالت|ذهبت|كانت|فعلت)\b'),
             lambda m: m.group(0).replace('قالت', 'قالوا').replace('ذهبت', 'ذهبوا')
                              .replace('كانت', 'كانوا').replace('فعلت', 'فعلوا'),
             'عدم تطابق الفعل المؤنث مع جمع المذكر (هم)'),

            (re.compile(r'\bهن\s+(?:قال|ذهب|كان|فعل)\b'),
             lambda m: m.group(0).replace('قال', 'قلن').replace('ذهب', 'ذهبن')
                              .replace('كان', 'كن').replace('فعل', 'فعلن'),
             'عدم تطابق الفعل المذكر مع جمع المؤنث (هن)'),

            # =========== عدم تطابق الاسم والصفة ===========
            (re.compile(r'\bالأسد\s+(?:جميلة|كبيرة|صغيرة|سريعة)\b'),
             lambda m: m.group(0).replace('جميلة', 'جميل').replace('كبيرة', 'كبير')
                              .replace('صغيرة', 'صغير').replace('سريعة', 'سريع'),
             'عدم تطابق الصفة المؤنثة مع الاسم المذكر'),

            (re.compile(r'\bالشمس\s+(?:جميل|كبير|صغير|سريع)\b'),
             lambda m: m.group(0).replace('جميل', 'جميلة').replace('كبير', 'كبيرة')
                              .replace('صغير', 'صغيرة').replace('سريع', 'سريعة'),
             'عدم تطابق الصفة المذكرة مع الاسم المؤنث'),

            # =========== أخطاء شائعة في الكلمات ===========
            (re.compile(r'\bإذا\s+غير\s+موجود'),
             'إذا لم يوجد',
             'صيغة خاطئة: استخدم (إذا لم يوجد) بدل (إذا غير موجود)'),

            (re.compile(r'\bرفع\s+العمود\b'),
             'ارتفاع العمود',
             'صيغة خاطئة في السياق'),

            (re.compile(r'\bيجب\s+أن\s+نجب\b'),
             'يجب أن نجيب',
             'تصحيح نطقي'),

            # =========== أخطاء في صيغة المصدر ===========
            (re.compile(r'\bالقيام\s+ب(?:يقوم|تقوم)\b'),
             lambda m: 'القيام به' if 'ب' in m.group(0) else m.group(0),
             'تصحيح حرف الجر'),

            # =========== الاسم والعدد ===========
            (re.compile(r'\b3\s+(?:رجل|امرأة|طفل)\s+(?:هو|هي)\b'),
             lambda m: m.group(0).replace(' هو', ' هم').replace(' هي', ' هن'),
             'عدم تطابق العدد مع الضمير'),
        ]

    def fix_text(self, text: str) -> Tuple[str, List[str]]:
        """Fix Arabic grammar errors in text.
        
        Returns:
            Tuple of (fixed_text, list_of_fixes_applied)
        """
        fixed_text = text
        fixes_applied = []

        for pattern, replacement, description in self.rules:
            if isinstance(replacement, str):
                # Direct string replacement
                new_text = pattern.sub(replacement, fixed_text)
            else:
                # Function-based replacement
                new_text = pattern.sub(replacement, fixed_text)

            if new_text != fixed_text:
                fixes_applied.append(description)
                fixed_text = new_text
                log.info(f"Grammar fix applied: {description}")

        return fixed_text, fixes_applied

    def validate_sentence(self, sentence: str) -> List[str]:
        """Validate a sentence and return list of potential issues."""
        issues = []

        # Check for common patterns that indicate problems
        if re.search(r'\b(هو|هي|هم|هن)\s+\w+[ةتن]\b', sentence):
            match = re.search(r'\b(هو|هي|هم|هن)\s+(\w+)', sentence)
            if match:
                pronoun, verb = match.groups()
                if pronoun == 'هو' and verb.endswith(('ت', 'ة')):
                    issues.append(f"عدم تطابق: الضمير '{pronoun}' مع فعل مؤنث '{verb}'")

        # Check for isolated words that are often errors
        if 'غير موجود' in sentence and 'إذا' in sentence:
            issues.append("قد تكون صيغة خاطئة: استخدم 'إذا لم يوجد' بدل 'إذا غير موجود'")

        return issues


def fix_arabic_grammar_batch(texts: List[str]) -> Dict[str, str]:
    """Fix grammar for multiple texts at once.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary mapping original text to fixed text
    """
    fixer = ArabicGrammarFixer()
    results = {}

    for text in texts:
        fixed, _ = fixer.fix_text(text)
        results[text] = fixed

    return results
