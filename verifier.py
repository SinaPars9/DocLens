import re
from typing import List, Dict, Any, Optional


class AnswerVerifier:
    def __init__(self) -> None:
        self.number_pattern: re.Pattern[str] = re.compile(
            r"(?:[-+]?[£$€]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:%|[KMB])?)|(?:\(\s?[£$€]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?\))"
        )

        self.math_words_pattern: re.Pattern[str] = re.compile(
            r"\b("
            r"difference|sum|total|totaled|average|mean|median|combined|count|amount|"
            r"multiply|multiplied|times|divide|divided|division|quotient|plus|minus|product|calculated|computed|derived|"
            r"rate|ratio|margin|percentage|percent|proportion|fraction|"
            r"compare[ds]?|comparison|roughly|approximately|approx"
            r")\b|[\+\-\*\/\÷\=\×]",
            re.IGNORECASE,
        )
        self.increase_pattern: re.Pattern[str] = re.compile(
            r"\b(increase[ds]?|rise|risen|rose|grew|growth|grow|up|higher|gain(?:ed)?|surge[ds]?)\b", re.IGNORECASE
        )
        self.decrease_pattern: re.Pattern[str] = re.compile(
            r"\b(decrease[ds]?|drop(?:ped)?|fall|fell|decline[ds]?|down|lower|loss|slump(?:ed)?)\b", re.IGNORECASE
        )

    def _normalize_number(self, raw: str) -> Optional[float]:
        clean = raw.replace(",", "").replace("£", "").replace("$", "").replace("€", "").strip()
        is_negative = False

        if clean.startswith("(") and clean.endswith(")"):
            is_negative = True
            clean = clean[1:-1].strip()

        multiplier = 1.0
        if clean.endswith("K") or clean.endswith("k"):
            multiplier = 1000.0
            clean = clean[:-1]
        elif clean.endswith("M") or clean.endswith("m"):
            multiplier = 1000000.0
            clean = clean[:-1]
        elif clean.endswith("B") or clean.endswith("b"):
            multiplier = 1000000000.0
            clean = clean[:-1]
        elif clean.endswith("%"):
            clean = clean[:-1]

        try:
            val = float(clean) * multiplier
            return -val if is_negative else val
        except ValueError:
            return None

    def _extract_numbers(self, text: str) -> List[float]:
        cleaned_text = re.sub(
            r"(?i)\b(page|table|figure|section|chapter|appendix)\s*\d+", "", text
        )
        nums: List[float] = []
        for m in self.number_pattern.finditer(cleaned_text):
            val = self._normalize_number(m.group())
            if val is not None:
                if 1900 <= val <= 2099 and val == int(val):
                    continue
                nums.append(val)
        return nums

    def _find_source_snippet(self, sentence: str, chunks: List[Dict[str, Any]]) -> Optional[str]:
        sent_nums = self._extract_numbers(sentence)
        sent_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower()))

        best_chunk_text: Optional[str] = None
        best_score = -1

        for chunk in chunks:
            text = str(chunk.get("text", ""))
            chunk_nums = self._extract_numbers(text)
            chunk_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()))

            matches = sum(1 for n in sent_nums if any(abs(n - cn) < 0.0001 for cn in chunk_nums))
            score = (matches * 10) + len(sent_words.intersection(chunk_words))

            if score > best_score and score > 0:
                best_score = score
                best_chunk_text = text

        if best_chunk_text:
            chunk_sentences = re.split(r"(?<=[.!?])\s+", best_chunk_text)
            for cs in chunk_sentences:
                if any(
                    abs(n - cn) < 0.0001
                    for n in sent_nums
                    for cn in self._extract_numbers(cs)
                ):
                    return cs.strip()
            return (
                best_chunk_text[:300] + "..."
                if len(best_chunk_text) > 300
                else best_chunk_text
            )
        return None

    def verify_answer(self, answer: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        claims: List[Dict[str, Any]] = []
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10
        ]

        for sent in sentences:
            sent_nums = self._extract_numbers(sent)
            has_math_word = bool(self.math_words_pattern.search(sent))
            has_increase = bool(self.increase_pattern.search(sent))
            has_decrease = bool(self.decrease_pattern.search(sent))

            source_snippet = self._find_source_snippet(sent, chunks)

            status = "⚠️ Check Manually"
            details = "Semantic check needed."

            if not sent_nums and not has_math_word:
                status = "✅ Grounded"
                details = "No specific numerical claims to verify."
                
            elif has_math_word:
                if source_snippet:
                    status = "✅ Calculated (Verified Operands)"
                    details = "Contains math, but base numbers found in source."
                else:
                    status = "❌ Hallucination"
                    details = "Calculated numbers do not match any source operands."
                    
            else:
                
                if len(sent_nums) >= 2 and (has_increase or has_decrease):
                    n1, n2 = sent_nums[0], sent_nums[1]
                    if has_increase and n2 < n1:
                        status = "⚠️ Contradicted"
                        details = f"Claimed increase, but {n2} is less than {n1}."
                    elif has_decrease and n2 > n1:
                        status = "⚠️ Contradicted"
                        details = f"Claimed decrease, but {n2} is greater than {n1}."
                    elif source_snippet:
                        status = "✅ Verified"
                        details = "Numbers match the source exactly."
                    else:
                        status = "❌ Hallucination"
                        details = "Claimed numbers not found in retrieved context."
                        
                
                elif source_snippet:
                    if len(sent_nums) > 1:
                        snippet_nums = self._extract_numbers(source_snippet)
                        all_found = all(any(abs(n - sn) < 0.0001 for sn in snippet_nums) for n in sent_nums)
                        if all_found:
                            status = "✅ Verified"
                            details = "All numbers match the source exactly."
                        else:
                            status = "🧮 Partially Supported"
                            details = "Some numbers match the source, but not all."
                    else:
                        status = "✅ Verified"
                        details = "Numbers match the source exactly."
                else:
                    status = "❌ Hallucination"
                    details = "Claimed numbers not found in retrieved context."

            claims.append(
                {
                    "sentence": sent,
                    "status": status,
                    "details": details,
                    "source_snippet": (
                        source_snippet if source_snippet else "No supporting text found."
                    ),
                    "numbers_found": sent_nums,
                }
            )

        return {
            "is_fully_verified": all(c["status"].startswith("✅") for c in claims),
            "claims": claims,
        }