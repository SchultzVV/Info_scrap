"""
Product Extractor - Classical CV pipeline for structured product info extraction

Endpoint: POST /product
Returns: title, price, oldPrice, disponivel, via_webhook, stock

Pipeline:
  1. ROI Detection  - Find the product info box using text-density contour analysis
  2. Price Detection - Classify promotional vs regular price with CV (strikethrough,
                       positional order, value comparison, OCR-artifact 'a$')
  3. Title Extraction - Longest high-scoring text line near the price region
  4. Stock/Availability - Parse quantity patterns from full OCR text
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL image to BGR numpy array."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _extract_price_value(text: str) -> Optional[float]:
    """
    Parse Brazilian price text to float.

    Examples:
        "R$ 4.475,00" -> 4475.0
        "4.475,00"    -> 4475.0
        "4475"        -> 4475.0
        "a$ 5.799,00" -> 5799.0  (OCR artifact for strikethrough price)
    """
    cleaned = re.sub(r'[RrAa]\$', '', text).strip()
    cleaned = re.sub(r'[a-zA-Z]', '', cleaned).strip()
    cleaned = cleaned.strip(".,")

    if not cleaned:
        return None

    if '.' in cleaned and ',' in cleaned:
        cleaned = cleaned.replace('.', '').replace(',', '.')
    elif ',' in cleaned:
        cleaned = cleaned.replace(',', '.')
    elif cleaned.count('.') > 1:
        cleaned = cleaned.replace('.', '')
    elif '.' in cleaned:
        parts = cleaned.split('.')
        if len(parts) == 2 and len(parts[1]) == 3:
            cleaned = cleaned.replace('.', '')

    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_price(value: Optional[float]) -> Optional[str]:
    """Format price as 'R$ NNNN.NN' (standard dot-decimal)."""
    if value is None:
        return None
    return f"R$ {value:.2f}"


# ---------------------------------------------------------------------------
# Main Extractor
# ---------------------------------------------------------------------------

class ProductExtractor:
    """
    Structured product information extractor.

    Pipeline:
    1. AdvancedImageAnalyzer – handles price ROI detection, strikethrough CV,
       'a$' OCR-artifact detection, and value-based old/current classification.
       This layer is already validated and working correctly.
    2. Title search – focused OCR on the region immediately ABOVE the price ROI;
       filters breadcrumbs ('>' patterns) and merges continuation lines ('-').
    3. Stock / availability – full-image OCR regex; handles 'Quantidade: 1
       unidade (+50 disponíveis)', 'Estoque disponível', 'Esgotado', etc.
    """

    # --- Regex patterns -------------------------------------------------------

    _PRICE_LINE_RE = re.compile(
        r'(?:[RrAa]\$|^\d{3,}[.,])',
    )

    _STOCK_PATTERNS = [
        # "quantidade: 1 unidade (+50 disponíveis)"
        re.compile(
            r'quantidade[:\s]+\d+\s*unidade[s]?\s*\(\+?\s*(\d+)\s*disponíve[il]s?\)',
            re.IGNORECASE,
        ),
        # "(+50 disponíveis)"
        re.compile(r'\(\+?\s*(\d+)\s*disponíve[il]s?\)', re.IGNORECASE),
        # "+50 disponíveis"
        re.compile(r'\+\s*(\d+)\s*disponíve[il]s?', re.IGNORECASE),
        # "50 disponíveis"
        re.compile(r'(\d+)\s*disponíve[il]s?', re.IGNORECASE),
        # "50 unidades disponíveis"
        re.compile(r'(\d+)\s*unidades?\s+disponíve[il]s?', re.IGNORECASE),
        # "somente 5 unid" / "últimas 3 unid"
        re.compile(r'(?:somente|últimas?)\s+(\d+)\s*unid', re.IGNORECASE),
        # "em estoque: 50"
        re.compile(r'estoque[:\s]+(\d+)', re.IGNORECASE),
    ]

    _UNAVAILABLE_RE = re.compile(
        r'\b(?:esgotado|indisponível|fora\s+de\s+estoque|sem\s+estoque|'
        r'out\s+of\s+stock|unavailable)\b',
        re.IGNORECASE,
    )

    # -------------------------------------------------------------------------

    def __init__(self):
        # Lazy imports to avoid circular dependency issues
        from analyzer import AdvancedImageAnalyzer
        from ecommerce_parser import EcommerceParser
        self._analyzer = AdvancedImageAnalyzer()
        self._parser = EcommerceParser()

    def extract(self, image_data: bytes) -> Dict:
        """
        Full extraction pipeline.

        Args:
            image_data: Raw image bytes (PNG/JPG).

        Returns:
            {title, price, oldPrice, disponivel, via_webhook, stock}
        """
        pil_img = Image.open(io.BytesIO(image_data))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # ── Step 1: Full image OCR ─────────────────────────────────────────────
        full_text = pytesseract.image_to_string(pil_img, lang="por+eng")
        lines = [l.strip() for l in full_text.split("\n") if l.strip()]

        # ── Step 2a: Find title first (anchor for price region) ────────────────
        title = self._find_title_using_heuristics(lines)

        # ── Step 2b: Price extraction (filter by region after title) ──────────
        current_price = None
        old_price = None
        if title:
            # Extract prices only after the title line
            title_idx = next((i for i, line in enumerate(lines) if title in line), None)
            if title_idx is not None:
                after_title_text = "\n".join(lines[title_idx:])
                prices = self._extract_main_prices(after_title_text, lines[title_idx:])
                
                # Assign prices based on scenario
                if len(prices) >= 2:
                    # Scenario 2 or 3: we have old and current price
                    old_price = prices[0]  # Higher value
                    current_price = prices[1]  # Lower value
                elif len(prices) == 1:
                    # Scenario 1 or 4: only one price or only installment total
                    current_price = prices[0]
        
        # Fallback: try full-text extraction if title-based didn't work
        if current_price is None:
            ep = self._parser.parse(full_text, lines).get("product", {})
            current_price = ep["current_price"]["value"] if ep.get("current_price") else None
            old_price     = ep["old_price"]["value"]     if ep.get("old_price")     else None

            # Fallback: AdvancedImageAnalyzer (strikethrough CV)
            if current_price is None:
                ap = self._analyzer.analyze_image(image_data).get("product", {})
                current_price = ap["current_price"]["value"] if ap.get("current_price") else None
                old_price     = ap["old_price"]["value"]     if ap.get("old_price")     else None

        # ── Step 3: Stock / availability ───────────────────────────────────
        stock = self._parse_stock(full_text)

        return {
            "title":      title,
            "price":      _format_price(current_price),
            "oldPrice":   _format_price(old_price),
            "disponivel": stock["disponivel"],
            "via_webhook": False,
            "stock":      stock["stock"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Title extraction  – improved heuristics (no price dependency)
    # ─────────────────────────────────────────────────────────────────────────

    # Navigation/UI words that never appear in product titles
    _NAV_MARKERS = [
        'buscar', 'seguidor', 'seguindo', 'seguir', 'apartir', 'para voc',
        'mais vendido', 'ofertas', 'cookies', 'privacidade', 'usamos',
        'configurar', 'aceitar', 'ptz', 'menu', 'carrinho', 'entrar',
        'cadastro', 'frete grátis', 'frete gratis', 'ir para produto',
        'sem juros', 'mercado pago', 'curadoria',
        # CEP / delivery modal
        'inclua seu cep', 'confira o envio', 'prazos de entrega',
        'custos e', 'mais tarde',
    ]

    def _find_title_using_heuristics(self, lines: List[str]) -> Optional[str]:
        """
        Find product title.

        Strategy:
        1. Anchor on the first substantial price line (R$ >= 100).
        2. Search ONLY in lines BEFORE that anchor (titles always appear above prices).
        3. Apply hard filters to remove nav/UI lines.
        4. Score survivors; merge continuation lines ending with '-'.

        Hard filters (any one disqualifies a line):
        - Contains 'R$' or 'a$'    → price line
        - Contains '>'              → breadcrumb
        - Contains '@'             → store badge / social
        - Matches a nav/UI marker  → navigation bar
        - Fewer than 10 alpha chars or fewer than 3 words
        """
        # Step 1: find the price anchor (first R$ line with value >= 100)
        price_anchor_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if re.search(r'R\$', line, re.IGNORECASE):
                val = _extract_price_value(line)
                if val and val >= 100:
                    price_anchor_idx = i
                    break

        # Search window: lines before the price anchor (or all lines if no anchor)
        search_range = lines[:price_anchor_idx] if price_anchor_idx else lines

        candidates = []
        for orig_idx, line in enumerate(search_range):
            line_lower = line.lower()

            # Hard filters
            if re.search(r'[Rr]\$|[Aa]\$', line):          # price line
                continue
            if '>' in line:                                   # breadcrumb
                continue
            if '@' in line:                                   # badge / social
                continue
            if any(m in line_lower for m in self._NAV_MARKERS):
                continue

            alpha_chars = [c for c in line if c.isalpha()]
            if len(alpha_chars) < 10:
                continue
            if len(line.split()) < 3:
                continue

            score = len(alpha_chars) + len(line.split()) * 2
            candidates.append((score, orig_idx, line))

        if not candidates:
            return None

        candidates.sort(reverse=True)
        best_score, best_idx, best = candidates[0]

        # Merge continuation line (ends with '-')
        if best.rstrip().endswith('-') and best_idx + 1 < len(lines):
            nxt = lines[best_idx + 1]
            if nxt and '@' not in nxt and not re.search(r'[Rr]\$', nxt):
                best = best.rstrip() + ' ' + nxt

        return best

    # ── Regex constants for price extraction ───────────────────────────────
    # Matches installment patterns – these are ALWAYS skipped:
    #   "21x R$ 185,44"  "12x R$ 372"  "12 x R$ 372,50"
    _INSTALLMENT_RE = re.compile(
        r'\d+\s*[xX]\s*(?:R\$|[Rr]?\$)\s*\d+(?:[,\.]\d{1,2})?',
        re.IGNORECASE,
    )
    # Matches split OCR installment: "xR 21728"  "xR$ 21728"
    _BROKEN_INST_RE = re.compile(r'[xX]\s*[Rr]\$?\s*(\d{4,6})')

    # Standalone R$ price (not part of an installment already handled above)
    # Uses \d+ (greedy) so "R$5799" captures "5799" not just "579"
    _STANDALONE_PRICE_RE = re.compile(
        r'R\$\s*(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)',
        re.IGNORECASE,
    )
    # OCR artifact for strikethrough price: "a$5709"  "a$ 5.799"
    # Uses \d+ so "a$5709" captures "5709" not "570"
    _ARTIFACT_PRICE_RE = re.compile(
        r'[Aa]\$\s*(\d+(?:[.,]\d{3})*(?:[.,]\d{2})?)',
    )
    # Orphan number on its own line: "816728," → old price
    _ORPHAN_NUM_RE = re.compile(r'^(\d{4,7})[,\.]?\s*$')

    def _extract_main_prices(self, text: str, lines: List[str]) -> List[float]:
        """
        Extract product prices with these strict rules:

        1. Installment values (Nx R$) are ALWAYS ignored — never become a price.
        2. Standalone R$ values that are NOT part of an installment → main prices.
        3. 'a$XXXX' OCR artifact (strikethrough) → old price candidate.
        4. Orphan number on its own line (e.g. '816728,') → old price candidate.
        5. If 2+ main prices: higher = old, lower = current.
        6. If 1 main price: current only (no old price).
        """
        full_text = "\n".join(lines) if isinstance(lines, list) else text
        seen: set = set()
        main_prices: List[float] = []
        old_candidates: List[float] = []   # from artifacts / orphan numbers

        # ── Step 1: mark installment spans to exclude them ───────────────────
        installment_spans: List[Tuple[int, int]] = []
        for m in self._INSTALLMENT_RE.finditer(full_text):
            installment_spans.append((m.start(), m.end()))
        # Also mark broken OCR installment spans ("xR 21728")
        for m in self._BROKEN_INST_RE.finditer(full_text):
            installment_spans.append((m.start(), m.end()))

        def _in_installment(pos: int) -> bool:
            return any(s <= pos < e for s, e in installment_spans)

        # ── Step 2: collect standalone R$ prices ────────────────────────────
        for m in self._STANDALONE_PRICE_RE.finditer(full_text):
            if _in_installment(m.start()):
                continue
            val = _extract_price_value(f"R${m.group(1)}")
            if val and 100 <= val <= 100_000 and val not in seen:
                seen.add(val)
                main_prices.append(val)

        # ── Step 3: collect OCR artifact 'a$XXXX' as old price candidate ────
        for m in self._ARTIFACT_PRICE_RE.finditer(full_text):
            val = _extract_price_value(f"R${m.group(1)}")
            if val and 100 <= val <= 100_000 and val not in seen:
                seen.add(val)
                old_candidates.append(val)

        # ── Step 4: collect orphan numbers as old price candidate ────────────
        for line in lines if isinstance(lines, list) else full_text.split('\n'):
            m = self._ORPHAN_NUM_RE.match(line.strip())
            if not m:
                continue
            raw = m.group(1)
            # Try interpretation: remove leading digit if 6-7 chars → XXXX.XX
            for candidate_str in [
                raw[1:] if len(raw) >= 6 else None,  # strip leading noise digit
                raw,
            ]:
                if not candidate_str or len(candidate_str) < 4:
                    continue
                parsed = candidate_str[:-2] + '.' + candidate_str[-2:]
                try:
                    val = float(parsed)
                except ValueError:
                    continue
                if 50 <= val <= 100_000 and val not in seen:
                    seen.add(val)
                    old_candidates.append(val)
                    break

        # ── Step 5: resolve scenarios ─────────────────────────────────────
        main_prices.sort(reverse=True)   # highest first

        if len(main_prices) >= 2:
            # Two explicit R$ prices → higher = old, lower = current
            return [main_prices[0], main_prices[1]]

        if len(main_prices) == 1 and old_candidates:
            # One R$ price + artifact/orphan → artifact is old price
            best_old = max(old_candidates)
            if best_old > main_prices[0]:
                return [best_old, main_prices[0]]

        return main_prices   # [current] or []

    # Price pattern used to identify the price anchor line
    _PRICE_RE = re.compile(
        r'R\$\s*\d|\bR\$\d|\d{3,}[.,]\d{2}|[Aa]\$\s*\d',
    )

    def _find_title_near_price_in_text(
        self,
        lines: List[str],
        current_price: Optional[float],
        old_price: Optional[float],
    ) -> Optional[str]:
        """
        Find the product title by scanning OCR lines just BEFORE the first
        price line.

        Reading-order insight (Mercado Livre screenshot):
          breadcrumb  → skip ('>' in short line)
          store badge → skip
          [PRODUCT TITLE line 1]    ← target  (closest text-line above prices)
          [PRODUCT TITLE line 2?]   ← merge if line 1 ends w/ '-'
          rating / count            → skip
          R$5799   ← price anchor   (first price line in OCR output)

        Strategy:
          1. Find the first line index where a price token appears.
          2. Look at the window [i-8 .. i-1].
          3. Score each candidate (len*1.5 + alpha_chars*0.5).
          4. Best candidate that is NOT a breadcrumb is the title.
          5. If it ends with '-', merge with the next line.
        """
        # Find the first line that contains any R$ price
        price_anchor_idx: Optional[int] = None
        for i, line in enumerate(lines):
            if self._PRICE_RE.search(line):
                # Confirm it actually contains a substantial price value
                val = _extract_price_value(line)
                if val and val >= 200:
                    price_anchor_idx = i
                    break

        # Search window: up to 8 lines before the price anchor
        if price_anchor_idx is None:
            search_lines = lines  # no price found, use all lines
        else:
            start = max(0, price_anchor_idx - 8)
            search_lines_with_idx = [
                (orig_idx, line)
                for orig_idx, line in enumerate(lines)
                if start <= orig_idx < price_anchor_idx
            ]

        candidates = []

        if price_anchor_idx is not None:
            window = search_lines_with_idx
        else:
            window = list(enumerate(lines))

        for orig_idx, line in window:
            if not line or not re.search(r"[a-zA-ZÀ-ú]", line):
                continue
            if len(line) < 10:
                continue
            # Skip breadcrumbs
            if ">" in line and len(line) < 100:
                continue
            # Skip store badge / social link lines (contain lone '@')
            if "@" in line:
                continue
            # Skip pure-rating / numeric lines (e.g. "4.9 ★ (12345)")
            alpha = re.sub(r"[\d.,% ()\-+|/$°º★☆*\[\]#@]", "", line).strip()
            if len(alpha) < 5:
                continue

            continues = line.rstrip().endswith(("-", "("))

            # Score based on merged text (if continuation, merge first for fair scoring)
            scored_text = line
            if continues and orig_idx + 1 < len(lines):
                nxt = lines[orig_idx + 1]
                if nxt and re.search(r"[a-zA-ZÀ-ú]", nxt) and "@" not in nxt:
                    scored_text = line.rstrip() + " " + nxt

            alpha_full = re.sub(r"[\d.,% ()\-+|/$°º★☆*\[\]#@]", "", scored_text).strip()
            score = len(scored_text) * 1.5 + len(alpha_full) * 0.5
            candidates.append({
                "text":     line,
                "score":    score,
                "idx":      orig_idx,
                "continues": continues,
            })

        if not candidates:
            return None

        best = max(candidates, key=lambda c: c["score"])

        # Merge continuation line (title ending with '-')
        if best["continues"]:
            nxt_idx = best["idx"] + 1
            if nxt_idx < len(lines):
                nxt = lines[nxt_idx]
                if nxt and re.search(r"[a-zA-ZÀ-ú]", nxt):
                    best["text"] = best["text"].rstrip() + " " + nxt

        return best["text"]

    # ─────────────────────────────────────────────────────────────────────────
    # ROI Detection  (price-anchored)
    # ─────────────────────────────────────────────────────────────────────────

    def _find_price_anchors(
        self, pil_img: Image.Image
    ) -> List[Dict]:
        """
        Downsampled full-image OCR to find approximate positions of major
        price tokens (R$XXXX, a$XXXX, XXXX,XX patterns).

        Returns list of {x, y, w, h, value} in *original* pixel coordinates.
        """
        SCALE = 0.4
        thumb = pil_img.resize(
            (int(pil_img.width * SCALE), int(pil_img.height * SCALE)),
            Image.LANCZOS,
        )
        data = pytesseract.image_to_data(
            thumb, lang="por+eng", config="--psm 3",
            output_type=pytesseract.Output.DICT,
        )
        anchors = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if conf < 15 or not text:
                continue
            if not re.search(r'(?:[RrAa]\$|\d{3,}[.,]\d{2}|\d{4,})', text):
                continue
            val = _extract_price_value(text)
            if val is None or val < 200:
                continue
            anchors.append({
                "x": int(data["left"][i] / SCALE),
                "y": int(data["top"][i] / SCALE),
                "w": int(data["width"][i] / SCALE),
                "h": int(data["height"][i] / SCALE),
                "value": val,
            })
        return anchors

    def _build_roi(
        self,
        anchors: List[Dict],
        img_w: int,
        img_h: int,
    ) -> Tuple[int, int, int, int]:
        """
        Build product info-panel ROI from price anchors.

        The product info panel is the rightmost price cluster.
        ROI extends 350 px above (room for title) and 300 px below (stock +
        installment info).

        Fallback: right 55 % of image if no anchors found.
        """
        if not anchors:
            rx = int(img_w * 0.45)
            return rx, 0, img_w - rx, img_h

        # Keep anchors in the rightmost cluster (within 350 px of rightmost X)
        ref_x = max(a["x"] for a in anchors)
        cluster = [a for a in anchors if abs(a["x"] - ref_x) < 350]
        if not cluster:
            cluster = anchors

        cx_min = min(a["x"] for a in cluster)
        cx_max = max(a["x"] + a["w"] for a in cluster)
        cy_min = min(a["y"] for a in cluster)
        cy_max = max(a["y"] + a["h"] for a in cluster)

        rx = max(0, cx_min - 50)
        ry = max(0, cy_min - 350)   # 350 px above first price for title
        rw = min(img_w - rx, (cx_max + 80) - rx)
        rh = min(img_h - ry, (cy_max + 300) - ry)
        return int(rx), int(ry), int(rw), int(rh)

    def _first_price_y(self, line_groups: List[List[Dict]]) -> Optional[int]:
        """Return the ROI-relative Y of the first word that is a major price."""
        for line in line_groups:
            for word in line:
                val = _extract_price_value(word["text"])
                if val is not None and val >= 200:
                    return word["y"]
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # OCR helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _get_word_data(
        self, img: np.ndarray, scale: float = 1.0
    ) -> List[Dict]:
        """
        Run Tesseract image_to_data and return cleaned word list.
        Coordinates are scaled back to original ROI space.
        """
        pil = _bgr_to_pil(img)
        data = pytesseract.image_to_data(
            pil,
            lang="por+eng",
            config="--psm 3",
            output_type=pytesseract.Output.DICT,
        )

        words = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not text or conf < 15:
                continue

            words.append(
                {
                    "text": text,
                    "x": int(data["left"][i] / scale),
                    "y": int(data["top"][i] / scale),
                    "w": int(data["width"][i] / scale),
                    "h": int(data["height"][i] / scale),
                    "conf": conf,
                    "block_num": data["block_num"][i],
                    "par_num": data["par_num"][i],
                    "line_num": data["line_num"][i],
                }
            )

        return words

    def _group_into_lines(
        self, words: List[Dict]
    ) -> List[List[Dict]]:
        """Group word-dicts by (block, paragraph, line) key; sort by Y."""
        lines: Dict[Tuple, List[Dict]] = {}
        for w in words:
            key = (w["block_num"], w["par_num"], w["line_num"])
            lines.setdefault(key, []).append(w)

        # Sort lines by their average Y coordinate
        sorted_lines = sorted(
            lines.values(), key=lambda ln: sum(w["y"] for w in ln) / len(ln)
        )
        # Sort words within each line by X
        return [sorted(ln, key=lambda w: w["x"]) for ln in sorted_lines]

    @staticmethod
    def _line_text(line: List[Dict]) -> str:
        return " ".join(w["text"] for w in line)

    # ─────────────────────────────────────────────────────────────────────────
    # Strikethrough Detection
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_strikethrough(
        self, roi_img: np.ndarray, word: Dict, scale: float = 1.0
    ) -> bool:
        """
        Detect horizontal strikethrough line through a word's bounding box.

        Method:
        - Extract pixel region for the word
        - Invert-threshold to isolate dark ink
        - Compute horizontal projection (sum per row) in the MIDDLE 30-70 % of height
        - If any row covers > 45 % of width → strikethrough present
        """
        try:
            x = int(word["x"] * scale)
            y = int(word["y"] * scale)
            w = int(word["w"] * scale)
            h = int(word["h"] * scale)

            if w < 4 or h < 4:
                return False

            region = roi_img[
                max(0, y): min(roi_img.shape[0], y + h),
                max(0, x): min(roi_img.shape[1], x + w),
            ]
            if region.size == 0:
                return False

            gray = (
                cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                if region.ndim == 3
                else region
            )
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            mid_s = int(h * 0.30)
            mid_e = int(h * 0.70)
            section = binary[mid_s:mid_e, :]
            if section.size == 0:
                return False

            proj = np.sum(section, axis=1) / (w * 255 + 1e-9)
            return float(np.max(proj)) > 0.45

        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Price Classification
    # ─────────────────────────────────────────────────────────────────────────

    def _classify_prices(
        self,
        line_groups: List[List[Dict]],
        roi_img: np.ndarray,
        scale: float = 1.0,
    ) -> Dict:
        """
        Multi-signal price classification:

        Signal A – OCR artifact 'a$' / 'A$' (Tesseract reads strikethrough R$ as a$)
        Signal B – Strikethrough detected visually on the word bounding box
        Signal C – Vertical position: old price typically appears ABOVE current price
        Signal D – Value comparison: old_price > current_price (always)

        Returns: {current_price: float|None, old_price: float|None}
        """

        # ── 1. Collect all price candidates from every line ──────────────────
        candidates = []  # {value, y, avg_h, has_strike, is_a_dollar, line_text}

        for line in line_groups:
            line_text = self._line_text(line)

            # Quick skip if line has no price-like content
            if not re.search(r'\d{2,}', line_text):
                continue

            # Check a$ artifact at line level
            line_has_a_dollar = bool(re.search(r'\b[aA]\$', line_text))

            # Aggregate price values found in this line
            for word in line:
                val = _extract_price_value(word["text"])
                if val is None or val < 50:
                    continue

                has_strike = self._detect_strikethrough(roi_img, word, scale)

                candidates.append(
                    {
                        "value": val,
                        "y": word["y"],
                        "avg_h": word["h"],
                        "has_strike": has_strike,
                        "is_a_dollar": line_has_a_dollar,
                        "line_text": line_text,
                    }
                )

        if not candidates:
            return {"current_price": None, "old_price": None}

        # ── 2. Sort by vertical position (reading order top→bottom) ──────────
        candidates.sort(key=lambda c: c["y"])

        # ── 3. Deduplicate: keep first occurrence of each rounded value ───────
        seen: set = set()
        unique: List[Dict] = []
        for c in candidates:
            key = round(c["value"], -1)   # round to nearest 10
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # ── 4. Apply signals to classify ──────────────────────────────────────
        old_price: Optional[float] = None
        current_price: Optional[float] = None

        # Priority list of old-price signals
        old_by_signal = [
            c for c in unique if c["has_strike"] or c["is_a_dollar"]
        ]
        non_old = [c for c in unique if c not in old_by_signal]

        if old_by_signal:
            old_price = old_by_signal[0]["value"]

        if non_old:
            # Among remaining candidates, current price is the most prominent
            # Heuristic: largest font (highest avg_h) in top part of ROI
            non_old_sorted = sorted(non_old, key=lambda c: (-c["avg_h"], c["y"]))
            current_price = non_old_sorted[0]["value"]
        elif len(unique) >= 2 and old_price is None:
            # No signals detected → use value ordering (bigger = old)
            by_value = sorted(unique, key=lambda c: c["value"], reverse=True)
            old_price = by_value[0]["value"]
            current_price = by_value[1]["value"]
        elif len(unique) == 1:
            current_price = unique[0]["value"]

        # ── 5. Sanity-check: old must be greater than current ─────────────────
        if old_price is not None and current_price is not None:
            if old_price < current_price:
                old_price, current_price = current_price, old_price

        # ── 6. If we only got one price, zero out old ─────────────────────────
        if current_price is None and old_price is not None:
            current_price = old_price
            old_price = None

        return {"current_price": current_price, "old_price": old_price}

    # ─────────────────────────────────────────────────────────────────────────
    # Title Extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _find_title(
        self,
        line_groups: List[List[Dict]],
        first_price_y: Optional[int],
    ) -> Optional[str]:
        """
        Extract product title from OCR lines ABOVE the first price.

        Filters:
        - Must have letters
        - Length >= 10 chars
        - Not a breadcrumb (contains '>' AND < 100 chars)
        - Not a pure price/numeric line

        Scoring: len(text)*1.5 + avg_word_height*2

        Multi-line merge: if the best candidate line ends with '-' or '(',
        it is joined with the immediately following line (continuation).
        """
        upper_limit = first_price_y if first_price_y is not None else float("inf")

        candidates = []
        for i, line in enumerate(line_groups):
            avg_y = sum(w["y"] for w in line) / len(line)
            if avg_y >= upper_limit:
                break  # stop at first price line

            text = self._line_text(line)
            if not text or not re.search(r"[a-zA-ZÀ-ú]", text):
                continue
            if len(text) < 10:
                continue
            # Filter breadcrumb navigation lines  ("A > B > C")
            if ">" in text and len(text) < 100:
                continue
            # Filter pure price/numeric lines
            stripped = re.sub(r"[R$\d.,% ()\-+xX°º]", "", text).strip()
            if len(stripped) < 4:
                continue

            avg_h = sum(w["h"] for w in line) / len(line)
            score = len(text) * 1.5 + avg_h * 2.0
            candidates.append({
                "text": text, "score": score, "y": avg_y,
                "idx": i,
                "ends_incomplete": text.rstrip().endswith(("-", "(")),
            })

        if not candidates:
            return None

        best = max(candidates, key=lambda c: c["score"])

        # Merge continuation line when title ends with '-'
        if best["ends_incomplete"]:
            next_idx = best["idx"] + 1
            if next_idx < len(line_groups):
                nxt = line_groups[next_idx]
                nxt_y = sum(w["y"] for w in nxt) / len(nxt)
                if nxt_y < upper_limit and abs(nxt_y - best["y"]) < 60:
                    nxt_text = self._line_text(nxt)
                    if nxt_text and re.search(r"[a-zA-ZÀ-ú]", nxt_text):
                        best["text"] = best["text"].rstrip() + " " + nxt_text

        return best["text"]

    # ─────────────────────────────────────────────────────────────────────────
    # Stock & Availability
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_stock(self, text: str) -> Dict:
        """
        Parse stock quantity and availability flag from OCR text.

        Priority:
        1. Unavailability keywords → disponivel=False
        2. Most-specific stock pattern first (quantity selector with count)
        3. Default: disponivel=True, stock=None
        """
        if self._UNAVAILABLE_RE.search(text):
            return {"disponivel": False, "stock": None}

        for pattern in self._STOCK_PATTERNS:
            match = pattern.search(text)
            if match:
                try:
                    return {"disponivel": True, "stock": int(match.group(1))}
                except (IndexError, ValueError):
                    return {"disponivel": True, "stock": None}

        return {"disponivel": True, "stock": None}
