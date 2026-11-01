"""Text cleaning shared by training and prediction.

The ISOT dataset has "giveaways" that tell the model which file an article
came from instead of what it says. Phase 1 learned these shortcuts:

- almost every real article starts with "WASHINGTON (Reuters) - "
- many fake articles end with "Featured image via ..." credits
- fake articles write contractions as "don t", real ones as "don’t"
- fake articles contain more links and @twitter handles

clean_text() removes these so the model has to learn from the writing itself.
"""

import re

# "WASHINGTON (Reuters) - " / "LONDON/PARIS (Reuters) - " at the start
DATELINE = re.compile(r"^.{0,120}?\(reuters\)\s*-\s*")
# image credits like "Featured image via Getty Images"
IMAGE_CREDIT = re.compile(r"(featured )?(image|photo|picture)s? (via|by|from|credit)\b.*?(\.|$)")
URL = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+|\b\S+\.(com|org|net)\S*")
HANDLE = re.compile(r"@\w+")
APOSTROPHE = re.compile(r"[’‘'`]")
# "don t" -> "dont", "trump s" -> "trumps" so both files end up the same
SPLIT_CONTRACTION = re.compile(r"\b([a-z]+) (t|s|re|ve|ll|d|m)(?=[\s.,!?;:\"]|$)")
SOURCE_WORDS = re.compile(r"\b(reuters|21st century wire|century wire)\b")


def clean_text(text):
    text = str(text).lower()
    text = DATELINE.sub("", text)
    text = IMAGE_CREDIT.sub(" ", text)
    text = URL.sub(" ", text)
    text = HANDLE.sub(" ", text)
    text = SOURCE_WORDS.sub(" ", text)
    text = APOSTROPHE.sub("", text)
    # join contractions before hyphens are removed, so "re-election" stays apart
    text = SPLIT_CONTRACTION.sub(r"\1\2", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
