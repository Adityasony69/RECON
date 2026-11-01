from preprocess import clean_text


def test_removes_reuters_dateline():
    assert clean_text("WASHINGTON (Reuters) - The vote failed.") == "the vote failed"


def test_contractions_same_in_both_styles():
    # Fake.csv writes "don t", True.csv writes "don’t" - both must become the same word
    assert clean_text("I don t know") == clean_text("I don’t know") == "i dont know"


def test_keeps_hyphenated_words_apart():
    assert clean_text("the re-election") == "the re election"


def test_removes_image_credits_links_and_handles():
    text = "Big news @someone https://t.co/abc. Featured image via Getty Images"
    assert clean_text(text) == "big news"


def test_handles_non_string():
    assert clean_text(None) == "none"
    assert clean_text(123) == ""
