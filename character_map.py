# Turkish to Arabic character mapping
turkish_to_arabic = {
    'ç': 'تش',  # Turkish 'ch' sound
    'ğ': 'غ',   # Soft g, approximated to Arabic 'gh'
    'ı': 'ى',   # Dotless i, approximated to Arabic alif maqsura
    'ö': 'و',   # Approximated to Arabic waw
    'ş': 'ش',   # Turkish 'sh' sound
    'ü': 'و',   # Approximated to Arabic waw
    'Ç': 'تش',
    'Ğ': 'غ',
    'İ': 'اي',  # Dotted capital I, approximated to Arabic alif + ya
    'Ö': 'و',
    'Ş': 'ش',
    'Ü': 'و',
}

# English to Arabic character mapping
english_to_arabic = {
    'a': 'ا',
    'b': 'ب',
    'c': 'ك',
    'd': 'د',
    'e': 'ي',
    'f': 'ف',
    'g': 'غ',
    'h': 'ه',
    'i': 'ي',
    'j': 'ج',
    'k': 'ك',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'o': 'و',
    'p': 'ب',
    'q': 'ك',
    'r': 'ر',
    's': 'س',
    't': 'ت',
    'u': 'و',
    'v': 'ف',
    'w': 'و',
    'x': 'كس',
    'y': 'ي',
    'z': 'ز',
}

# Add uppercase English letters
english_to_arabic.update({k.upper(): v for k, v in english_to_arabic.items()})

# Add any other specific character mappings here
other_mappings = {
    # Add more mappings as needed
}

# Combine all mappings into a single dictionary
char_map = {}
char_map.update(turkish_to_arabic)
char_map.update(english_to_arabic)
char_map.update(other_mappings)

def map_character(char):
    """Map a single character to its Arabic equivalent"""
    return char_map.get(char, char)

def map_text(text):
    """Map all characters in a text to their Arabic equivalents"""
    return ''.join(map_character(char) for char in text)