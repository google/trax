import six

from absl import logging


# Unicode utility functions that work with Python 2 and 3
def native_to_unicode(s):
    if is_unicode(s):
        return s
    try:
        return to_unicode(s)
    except UnicodeDecodeError:
        res = to_unicode(s, ignore_errors=True)
        logging.info("Ignoring Unicode error, outputting: %s", res)
        return res


def is_unicode(s):
    return isinstance(s, six.text_type)


def to_unicode(s, ignore_errors=False):
    if is_unicode(s):
        return s
    error_mode = "ignore" if ignore_errors else "strict"
    return s.decode("utf-8", errors=error_mode)


def to_unicode_ignore_errors(s):
    return to_unicode(s, ignore_errors=True)


def to_unicode_utf8(s):
    return s.decode("utf-8")


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def strip_ids(ids, ids_to_strip):
    """Strip ids_to_strip from the end IDs."""
    ids = list(ids)
    while ids and ids[-1] in ids_to_strip:
        ids.pop()
    return ids
