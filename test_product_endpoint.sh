#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# test_product_endpoint.sh
# Testa o endpoint POST /product em todos os 4 cenários conhecidos.
#
# Cenário 1 – primeioro_caso.png  | título + preço + parcelamento
# Cenário 2 – segundo_caso.png    | preço riscado + promo + parcelamento
# Cenário 3 – terceiro_caso.png   | preço riscado + atual + sem parcelamento
# Cenário 4 – iphone.png          | página completa (caso validado)
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL="${BASE_URL:-http://localhost:8000}"
PASS=0
FAIL=0

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

check() {
  local label="$1"
  local got="$2"
  local expected="$3"

  if [ "$got" = "$expected" ]; then
    printf "  ${GREEN}✔${NC} %-20s %s\n" "$label" "$got"
    ((PASS++))
  else
    printf "  ${RED}✘${NC} %-20s got=${RED}%s${NC}  expected=${GREEN}%s${NC}\n" \
      "$label" "$got" "$expected"
    ((FAIL++))
  fi
}

check_notnull() {
  local label="$1"
  local got="$2"

  if [ "$got" != "null" ] && [ -n "$got" ]; then
    printf "  ${GREEN}✔${NC} %-20s %s\n" "$label" "$got"
    ((PASS++))
  else
    printf "  ${RED}✘${NC} %-20s expected a value, got=${RED}null${NC}\n" "$label"
    ((FAIL++))
  fi
}

check_null() {
  local label="$1"
  local got="$2"

  if [ "$got" = "null" ]; then
    printf "  ${GREEN}✔${NC} %-20s null (correct)\n" "$label"
    ((PASS++))
  else
    printf "  ${YELLOW}~${NC} %-20s got=${YELLOW}%s${NC}  (expected null)\n" "$label" "$got"
    # warn only, not fail — informational
  fi
}

run_case() {
  local num="$1"
  local image="$2"
  local desc="$3"

  printf "\n${BOLD}${BLUE}━━━ Cenário %s: %s ━━━${NC}\n" "$num" "$desc"
  printf "    Imagem: %s\n" "$image"

  if [ ! -f "$image" ]; then
    printf "  ${RED}✘ Arquivo não encontrado: %s${NC}\n" "$image"
    ((FAIL++))
    return
  fi

  local response
  response=$(curl -s -X POST "$BASE_URL/product" -F "file=@$image")

  if [ -z "$response" ]; then
    printf "  ${RED}✘ Sem resposta do servidor${NC}\n"
    ((FAIL++))
    return
  fi

  # Pretty-print JSON
  echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
  echo ""

  # Extract fields
  local title price oldPrice disponivel stock
  title=$(echo "$response"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('title') or 'null')" 2>/dev/null)
  price=$(echo "$response"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('price') or 'null')" 2>/dev/null)
  oldPrice=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('oldPrice') or 'null')" 2>/dev/null)
  disponivel=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(str(d.get('disponivel')).lower())" 2>/dev/null)
  stock=$(echo "$response"    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('stock') if d.get('stock') is not None else 'null')" 2>/dev/null)

  echo "  Assertions:"
  # Run assertions passed to function (from arg 4 onward)
  shift 3
  while [ "$#" -gt 0 ]; do
    eval "$1"
    shift
  done
}

# ──────────────────────────────────────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────────────────────────────────────
printf "\n${BOLD}Verificando saúde da API...${NC} "
health=$(curl -s "$BASE_URL/health")
if echo "$health" | grep -q "healthy"; then
  printf "${GREEN}OK${NC}\n"
else
  printf "${RED}FALHOU${NC} — resposta: %s\n" "$health"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────────────
# Cenário 1: Smartphone Xiaomi — título + preço R$ 2.260 + parcelamento (OCR quebrado: xR 21728)
# ──────────────────────────────────────────────────────────────────────────────
run_case 1 "primeioro_caso.png" "Título + preço único + parcelamento" \
  'check_notnull "title"      "$title"' \
  'check         "price"      "$price"      "R$ 2260.00"' \
  'check_null    "oldPrice"   "$oldPrice"' \
  'check         "disponivel" "$disponivel" "true"'

# ──────────────────────────────────────────────────────────────────────────────
# Cenário 2: iPhone Branco — preço riscado (a$5709) + promo R$3.699 + parcelamento (21x R$185,44)
# ──────────────────────────────────────────────────────────────────────────────
run_case 2 "segundo_caso.png" "Preço riscado + promocional + parcelamento" \
  'check_notnull "title"      "$title"' \
  'check         "price"      "$price"      "R$ 3699.00"' \
  'check_notnull "oldPrice"   "$oldPrice"' \
  'check         "disponivel" "$disponivel" "true"'

# ──────────────────────────────────────────────────────────────────────────────
# Cenário 3: Omo 7L — preço riscado (816728 = R$167,28) + atual R$124 + sem parcelamento
# ──────────────────────────────────────────────────────────────────────────────
run_case 3 "terceiro_caso.png" "Preço riscado + atual + sem parcelamento" \
  'check_notnull "title"      "$title"' \
  'check         "price"      "$price"      "R$ 124.00"' \
  'check_notnull "oldPrice"   "$oldPrice"' \
  'check         "disponivel" "$disponivel" "true"'

# ──────────────────────────────────────────────────────────────────────────────
# Cenário 4: iPhone Preto — caso completo (validado anteriormente)
# ──────────────────────────────────────────────────────────────────────────────
run_case 4 "iphone.png" "Página completa — preço antigo + atual + parcelamento + estoque" \
  'check         "title"      "$title"      "iPhone 16e (128 GB) - Preto - Distribuidor Autorizado"' \
  'check         "price"      "$price"      "R$ 4475.00"' \
  'check         "oldPrice"   "$oldPrice"   "R$ 5799.00"' \
  'check         "disponivel" "$disponivel" "true"' \
  'check         "stock"      "$stock"      "50"'

# ──────────────────────────────────────────────────────────────────────────────
# Resumo
# ──────────────────────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL))
printf "\n${BOLD}━━━ Resultado: %d/%d passaram ${NC}" "$PASS" "$TOTAL"
if [ "$FAIL" -eq 0 ]; then
  printf "${GREEN}✔ Todos OK${NC}\n\n"
else
  printf "${RED}✘ %d falharam${NC}\n\n" "$FAIL"
fi
