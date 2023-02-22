def is_first_letter_capital(word: str) -> bool:
    if word[0].isupper():
        return True
    return False


def is_acronym(word: str) -> bool:
    if word.isupper():
        return True
    return False
