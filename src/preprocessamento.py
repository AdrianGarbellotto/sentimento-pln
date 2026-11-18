"""Normalização simples de textos em português para os experimentos de PLN."""

import re

# Lista própria de palavras funcionais comuns. Termos de negação como "não",
# "nunca" e "sem" foram deliberadamente preservados para não inverter o sentido.
STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas", "de", "do", "da",
    "dos", "das", "em", "no", "na", "nos", "nas", "por", "para", "com",
    "e", "ou", "que", "é", "foi", "ser", "ao", "aos", "se", "sua", "seu",
    "suas", "seus", "mais", "muito", "muita", "muitos", "muitas", "já",
    "também", "como", "mas", "me", "minha", "meu", "minhas", "meus",
    "lhe", "eles", "elas", "ele", "ela", "isso", "este", "esta", "esse",
    "essa", "aquele", "aquela", "porque", "quando", "onde", "até", "sobre",
}

_TOKEN_RE = re.compile(r"[a-záàâãéêíóôõúçü]+", flags=re.IGNORECASE)


def preprocessar_texto(texto, remover_stopwords=False):
    """Deixa o texto minúsculo, remove pontuação/números e opcionalmente stopwords."""
    if texto is None:
        texto = ""
    texto = str(texto).lower()
    tokens = _TOKEN_RE.findall(texto)
    if remover_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return " ".join(tokens)
