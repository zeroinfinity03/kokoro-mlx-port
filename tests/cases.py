"""What both models are run on.

CASES: single utterances for the stage-by-stage gates: (name, text, voice, speed).
TEXTS: narration, some of it long enough to be split into several chunks, for the end-to-end gate.
"""
LONG = ("When a list is already sorted, a binary search looks at the middle item first, compares it with the "
        "target, throws away the half that cannot contain it, and repeats the same step on what is left, so a "
        "million items need only about twenty comparisons, a billion need thirty, and the work grows so slowly "
        "that doubling the data adds a single extra step, which is why this simple idea sits underneath "
        "databases, dictionaries, and almost every search box you have ever typed into.")
CASES = [
    ("short_heart", "Hello world.", "af_heart", 1.0),
    ("medium_heart", "The quick brown fox jumps over the lazy dog, and then it runs back home before the rain begins.", "af_heart", 1.0),
    ("narration_michael", "A binary search halves the list at every step, so even a million items need only about twenty comparisons.", "am_michael", 1.0),
    ("long_heart", LONG, "af_heart", 1.0),
    ("british_fast", "The schedule for tomorrow's laboratory session has been changed to half past nine.", "bf_emma", 1.25),
    ("slow_bella", "Energy can change its form, but the total amount always stays the same.", "af_bella", 0.8),
    ("numbers_heart", "In 2024, Dr. Smith paid $3.50 for 12 apples; isn't that odd?", "af_heart", 1.0),
    ("question_adam", "Why does ice float on water, when most solids sink in their own liquid?", "am_adam", 1.1),
]

LIBRARY = ("Imagine a library with one million books, all in alphabetical order. You want one title. You could "
           "start at the first shelf and read every spine, and on a bad day that is a million looks. Or you could "
           "walk to the middle of the library, read one spine, and ask a single question: does my book come before "
           "this one, or after it? Whatever the answer, half the library is now out of the game, and you never have "
           "to look at it again. Do the same thing in the half that is left, and you are down to a quarter. Then an "
           "eighth. Then a sixteenth. After only twenty questions, one million books have become a single book, and "
           "it is the one you wanted. That is binary search. It is not clever because it is fast on a small shelf; it "
           "is clever because its work barely grows when the library does. Double the books to two million, and you "
           "need exactly one more question. A billion books need thirty. The only price is that the shelves must "
           "already be sorted, because the trick depends on knowing that everything to the left is smaller and "
           "everything to the right is larger.")
TEXTS = [
    ("phonebook", "af_heart", 1.0, "Imagine you have a phone book with one million names. To find one name, you could read every page from the start. That would take forever. A binary search does something smarter: it opens the book in the middle, checks which half the name must be in, and throws the other half away."),
    ("water", "am_michael", 1.0, "Water is strange. Most liquids shrink when they freeze, but water expands by about nine percent, which is why ice floats. If it did not, lakes would freeze from the bottom up, and almost nothing could live in them through the winter."),
    ("physics1905", "af_bella", 1.0, "In 1905, a 26-year-old patent clerk published four papers. One of them explained the photoelectric effect; another gave us E equals m c squared. Physics was never the same again."),
    ("british", "bf_emma", 1.0, "The schedule for tomorrow's laboratory session has been changed. Please bring your notebook, a pencil, and the results from last week's experiment on aluminium."),
    ("library_long", "af_heart", 1.0, LIBRARY),
    ("fast", "am_adam", 1.15, "A graph is just dots and lines. The dots are things, the lines are relationships, and almost any problem about connections can be drawn this way."),
    ("blend", "af_heart,af_bella", 0.9, "Two voices can be mixed into one. The result sits halfway between them."),
]
