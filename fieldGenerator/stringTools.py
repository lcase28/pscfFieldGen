""" Module containing useful functions for string parsing """
import re

def str_to_num(stringNum):
    """
    Convert string to either int or float with priority on int.
    
    Examples:
        "10.000" returns float
        "5" returns int
        "abc" raises error
        "1e25" returns float
    
    Parameters
    ----------
    stringNum : str
        String representing a numeric value
    
    Returns
    -------
    int if string represents a valid integer, otherwise float
    
    Raises
    ------
    ValueError : if stringNum does not represent a numeric value
    TypeError : if stringNum is not string, bytes-like object, or number
    """
    try:
        return int(stringNum)
    except(ValueError):
        return float(stringNum)

def wordsGenerator(stringIterable):
    """
    Split a string iterable into individual words.
    
    Words, here, defined as groups of characters separated by whitespace, 
    or groups of characters (including whitespace) fully enclosed by double-
    or single-quotes.
    
    example: words(["This is 'a test'", "'one word' oneword two word."])
        returns a generator producing: ["This", "is", "'a test'", "'one word'", "oneword", "two", "word."]
    
    Paramters
    ---------
    stringIterable : iterable of string objects
        The "stream" of strings to convert to words, such as a file.
    """ 
    lineStream = iter(stringIterable)
    splitPattern = re.compile("(\s+|\\\".*?\\\"|'.*?')")
    for line in lineStream:
        line = line.strip()
        #wordlist = [w for w in re.split("(\s+|\\\".*?\\\"|'.*?')", line) if w.strip()]
        wordlist = [w for w in splitPattern.split(line) if w.strip()]
        for word in wordlist:
            yield word.strip()
            

