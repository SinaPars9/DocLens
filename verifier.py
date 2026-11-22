import ast
import re
from typing import List, Dict, Any, Optional, Tuple


class AnswerVerifier:
    def __init__(self) -> None:
        self.number_pattern: re.Pattern[str] = re.compile(
            r"\b(?:[-+]?[£$€]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:%|[KMB])?)\b|(?<!\w)\(\s?[£$€]?\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s?\)(?!\w)"
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
            r"\b(increase[ds]?|rise|risen|rose|grew|growth|grow|up|higher|gain(?:ed)?|surge[ds]?)\b",
            re.IGNORECASE,
        )
        self.decrease_pattern: re.Pattern[str] = re.compile(
            r"\b(decrease[ds]?|drop(?:ped)?|fall|fell|decline[ds]?|down|lower|loss|slump(?:ed)?)\b",
            re.IGNORECASE,
        )

        # Symbolic equations, e.g. "2,400,000 + 1,416,000 = 3,816,000".
        # Runs on text normalized by _normalize_equation_text().
        self.equation_pattern: re.Pattern[str] = re.compile(
            r"((?:-?\d+(?:\.\d+)?)(?:\s*[+\-*/]\s*-?\d+(?:\.\d+)?)+)\s*=\s*(-?\d+(?:\.\d+)?)"
        )

        # Word-form equations, e.g. "the difference between A and B is C".
        self.word_equation_pattern: re.Pattern[str] = re.compile(
            r"\b(sum|difference|product|ratio|average|mean)\s+(?:of|between)\s+"
            r"(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)\s*=\s*(-?\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )

    # ------------------------------------------------------------------
    # Number handling
    # ------------------------------------------------------------------

    def _normalize_number(self, raw: str) -> Optional[float]:
        clean = (
            raw.replace(",", "")
            .replace("£", "")
            .replace("$", "")
            .replace("€", "")
            .strip()
        )
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

    @staticmethod
    def _close(a: float, b: float) -> bool:
        """Compare with a small tolerance to allow rounding (e.g. 14.3% vs 14.2857%)."""
        return abs(a - b) <= max(0.01, abs(b) * 0.005)

    @staticmethod
    def _format_number(value: float) -> str:
        if value == int(value):
            return f"{value:,.0f}"
        return f"{value:,.2f}"

    # ------------------------------------------------------------------
    # Arithmetic verification
    # ------------------------------------------------------------------

    def _normalize_equation_text(self, text: str) -> str:
        text = (
            text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
        )
        text = re.sub(r"(?i)\bmultiplied\s+by\b", "*", text)
        text = re.sub(r"(?i)\bdivided\s+by\b", "/", text)
        text = re.sub(r"(?i)\bplus\b", "+", text)
        text = re.sub(r"(?i)\bminus\b", "-", text)
        text = re.sub(r"(?i)\btimes\b", "*", text)
        text = re.sub(
            r"(?i)\b(?:equals?|equal\s+to|is\s+equal\s+to|results?\s+in|gives?|yields?)\b",
            "=",
            text,
        )
        # "the difference between A and B is C" -> "... = C"
        text = re.sub(r"(?i)\b(?:is|was|were|becomes?)\b", "=", text)
        text = re.sub(r"[£$€]", "", text)
        text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
        # Accounting-style "(1,234)" is treated as a plain positive number here;
        # _extract_numbers() keeps the negative-sign interpretation for claims.
        text = re.sub(r"(?<!\w)\((\d[\d,]*(?:\.\d+)?)\)", r"\1", text)
        return text

    def _eval_node(self, node: ast.AST) -> float:
        """Evaluate a whitelisted arithmetic AST without using eval()."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError("unsupported unary operator")

        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError
                return left / right
            raise ValueError("unsupported binary operator")

        raise ValueError("unsupported expression")

    def _eval_expression(self, expression: str) -> Optional[float]:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None

        try:
            return self._eval_node(tree.body)
        except (ValueError, ZeroDivisionError):
            return None

    def _word_candidates(self, kind: str, a: float, b: float) -> List[float]:
        kind = kind.lower()
        if kind == "sum":
            return [a + b]
        if kind == "difference":
            return [a - b, b - a]
        if kind == "product":
            return [a * b]
        if kind == "ratio":
            return [a / b] if b else []
        if kind in ("average", "mean"):
            return [(a + b) / 2]
        return []

    def _extract_equations(self, sentence: str) -> List[Tuple[str, List[float], float]]:
        """Return (expression, computed candidates, claimed result) tuples."""
        text = self._normalize_equation_text(sentence)
        equations: List[Tuple[str, List[float], float]] = []

        for match in self.equation_pattern.finditer(text):
            expression = match.group(1).strip()
            claimed = float(match.group(2))
            value = self._eval_expression(expression)
            if value is not None:
                equations.append((expression, [value], claimed))

        for match in self.word_equation_pattern.finditer(text):
            kind = match.group(1).lower()
            a = float(match.group(2))
            b = float(match.group(3))
            claimed = float(match.group(4))
            candidates = self._word_candidates(kind, a, b)
            if candidates:
                label = (
                    f"{kind} of {self._format_number(a)} and "
                    f"{self._format_number(b)}"
                )
                equations.append((label, candidates, claimed))

        return equations

    def _verify_equations(
        self, equations: List[Tuple[str, List[float], float]]
    ) -> Tuple[str, str]:
        checked: List[str] = []

        for expression, candidates, claimed in equations:
            if not any(self._close(candidate, claimed) for candidate in candidates):
                computed = candidates[0]
                return (
                    "⚠️ Contradicted",
                    f"Arithmetic check failed: {expression} = {computed:,.2f}, "
                    f"but the answer states {claimed:,.2f}.",
                )
            checked.append(f"{expression} = {claimed:,.2f}")

        return (
            "✅ Calculated (Verified)",
            "Arithmetic checked: " + "; ".join(checked) + ".",
        )

    def _source_numbers(self, chunks: List[Dict[str, Any]]) -> List[float]:
        numbers: List[float] = []
        for chunk in chunks:
            numbers.extend(self._extract_numbers(str(chunk.get("text", ""))))
        return numbers

    def _find_derivation(
        self, claimed: float, chunks: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Find a simple two-number operation on source numbers that yields the claim."""
        unique: List[float] = []
        seen = set()
        for number in self._source_numbers(chunks):
            if number not in seen:
                seen.add(number)
                unique.append(number)

        unique = unique[:60]

        for index, a in enumerate(unique):
            for b in unique[index + 1 :]:
                candidates: List[Tuple[str, float]] = [
                    (f"{self._format_number(a)} + {self._format_number(b)}", a + b),
                    (f"{self._format_number(a)} - {self._format_number(b)}", a - b),
                    (f"{self._format_number(b)} - {self._format_number(a)}", b - a),
                    (f"{self._format_number(a)} * {self._format_number(b)}", a * b),
                ]
                if b:
                    candidates.append(
                        (f"{self._format_number(a)} / {self._format_number(b)}", a / b)
                    )
                if a:
                    candidates.append(
                        (f"{self._format_number(b)} / {self._format_number(a)}", b / a)
                    )

                for label, value in candidates:
                    if self._close(value, claimed):
                        return f"{label} = {value:,.2f}"

        return None

    # ------------------------------------------------------------------
    # Source matching
    # ------------------------------------------------------------------

    def _find_source_snippet(
        self, sentence: str, chunks: List[Dict[str, Any]]
    ) -> Optional[str]:
        sent_nums = self._extract_numbers(sentence)
        sent_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower()))

        best_chunk_text: Optional[str] = None
        best_score = -1

        for chunk in chunks:
            text = str(chunk.get("text", ""))
            chunk_nums = self._extract_numbers(text)
            chunk_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()))

            matches = sum(
                1 for n in sent_nums if any(abs(n - cn) < 0.0001 for cn in chunk_nums)
            )
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

    # ------------------------------------------------------------------
    # Main verification
    # ------------------------------------------------------------------

    def verify_answer(
        self, answer: str, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        claims: List[Dict[str, Any]] = []
        sentences = [
            s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if len(s.strip()) > 10
        ]

        source_numbers = self._source_numbers(chunks)

        for sent in sentences:
            sent_nums = self._extract_numbers(sent)
            has_math_word = bool(self.math_words_pattern.search(sent))
            has_increase = bool(self.increase_pattern.search(sent))
            has_decrease = bool(self.decrease_pattern.search(sent))

            source_snippet = self._find_source_snippet(sent, chunks)
            equations = self._extract_equations(sent)

            status = "⚠️ Check Manually"
            details = "Semantic check needed."

            if not sent_nums and not has_math_word:
                status = "✅ Grounded"
                details = "No specific numerical claims to verify."

            elif equations:
                status, details = self._verify_equations(equations)

            elif has_math_word:
                direct_match = bool(sent_nums) and all(
                    any(self._close(n, sn) for sn in source_numbers) for n in sent_nums
                )

                if direct_match:
                    status = "✅ Verified"
                    details = "Numbers match the source exactly."
                elif sent_nums:
                    derivation = self._find_derivation(sent_nums[-1], chunks)
                    if derivation:
                        status = "✅ Calculated (Verified Operands)"
                        details = (
                            "Stated result is derivable from source operands: "
                            f"{derivation}."
                        )
                    else:
                        status = "❌ Hallucination"
                        details = (
                            "Calculated result cannot be produced from the source numbers."
                        )
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
                        all_found = all(
                            any(abs(n - sn) < 0.0001 for sn in snippet_nums)
                            for n in sent_nums
                        )
                        if all_found:
                            status = "✅ Verified"
                            details = "All numbers match the source exactly."
                        else:
                            derivation = self._find_derivation(sent_nums[-1], chunks)
                            if derivation:
                                status = "✅ Calculated (Verified Operands)"
                                details = (
                                    "Stated result is derivable from source operands: "
                                    f"{derivation}."
                                )
                            else:
                                status = "🧮 Partially Supported"
                                details = "Some numbers match the source, but not all."
                    else:
                        only_number = sent_nums[0]
                        direct_match = any(
                            self._close(only_number, sn) for sn in source_numbers
                        )
                        if direct_match:
                            status = "✅ Verified"
                            details = "Numbers match the source exactly."
                        else:
                            derivation = self._find_derivation(only_number, chunks)
                            if derivation:
                                status = "✅ Calculated (Verified Operands)"
                                details = (
                                    "Stated result is derivable from source operands: "
                                    f"{derivation}."
                                )
                            else:
                                status = "🧮 Partially Supported"
                                details = (
                                    "Number is not directly in the source and could not "
                                    "be derived from source operands."
                                )

                else:
                    derivation = (
                        self._find_derivation(sent_nums[-1], chunks)
                        if sent_nums
                        else None
                    )
                    if derivation:
                        status = "✅ Calculated (Verified Operands)"
                        details = (
                            "Stated result is derivable from source operands: "
                            f"{derivation}."
                        )
                    else:
                        status = "❌ Hallucination"
                        details = "Claimed numbers not found in retrieved context."

            claims.append(
                {
                    "sentence": sent,
                    "status": status,
                    "details": details,
                    "source_snippet": (
                        source_snippet
                        if source_snippet
                        else "No supporting text found."
                    ),
                    "numbers_found": sent_nums,
                }
            )

        return {
            "is_fully_verified": all(c["status"].startswith("✅") for c in claims),
            "claims": claims,
        }
