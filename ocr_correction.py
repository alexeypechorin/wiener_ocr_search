import os

from symspellpy.symspellpy import SymSpell, Verbosity  # import the module

def correct_ocr_errors(input_term, dictionary_path):
    sym_spell = load_symspell(dictionary_path)
    correct(input_term, sym_spell)

def load_symspell(dictionary_path):
    # create object
    initial_capacity = 83000
    # maximum edit distance per dictionary precalculation
    max_edit_distance_dictionary = 2
    prefix_length = 7
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,
                         prefix_length)

    # load dictionary
    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file
    print('loading dictionary...')
    if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
        print("Dictionary file not found")
        return

    return sym_spell

def correct(input_term, sym_spell):
    if isinstance(input_term, str):
        input_term = [input_term]

    # lookup suggestions for single-word input strings
    # input_term = "memebers"  # misspelling of "members"
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_edit_distance_dictionary)

    for term in input_term:
        max_edit_distance_lookup = 2
        suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
        suggestions = sym_spell.lookup(term, suggestion_verbosity,
                                       max_edit_distance_lookup)
        # display suggestion term, term frequency, and edit distance
        for suggestion in suggestions:
            print("{}, {}, {}".format(suggestion.term, suggestion.count,
                                      suggestion.distance))

        if len(suggestions) > 0: return suggestions[0].term
        else: return ''

    # # lookup suggestions for multi-word input strings (supports compound
    # # splitting & merging)
    # input_term = ("whereis th elove hehad dated forImuch of thepast who "
    #               "couqdn'tread in sixtgrade and ins pired him")
    # # max edit distance per lookup (per single word, not per whole input string)
    # max_edit_distance_lookup = 2
    # suggestions = sym_spell.lookup_compound(input_term,
    #                                         max_edit_distance_lookup)
    # # display suggestion term, edit distance, and term frequency
    # for suggestion in suggestions:
    #     print("{}, {}, {}".format(suggestion.term, suggestion.count,
    #                               suggestion.distance))

if __name__ == "__main__":
    correct_ocr_errors(input_term='memebers', dictionary_path=os.path.join('data', 'frequency_dictionary_en_82_765.txt'))