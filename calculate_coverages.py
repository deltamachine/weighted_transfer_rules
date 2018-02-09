import os
import re
import sys
from subprocess import check_output
from optparse import OptionParser, OptionGroup
from time import clock
from streamparser import parse


try: # see if lxml is installed
    from lxml import etree as ET
    if __name__ == "__main__":
        print("Using lxml library happily ever after.")
except ImportError: # it is not
    import xml.etree.ElementTree as ET
    if __name__ == "__main__":
        print("lxml library not found. Falling back to xml.etree,\n"
              "though it's highly recommended that you install lxml\n"
              "as it works dramatically faster than xml.etree.")


# regex lines to build up rexes for cat-items
any_tag_re = '<[a-z0-9-]+>'
any_num_of_any_tags_re = '({})*'.format(any_tag_re)

# apertium token (anything between ^ and $)
apertium_token_re = re.compile(r'\^(.*?)\$')


def cat_item_to_re(cat_item):
    """
    Get a pattern as specified in xml.
    Output a regex line that matches what 
    is specified by the pattern.

    Attention: ^ and $ here are NOT Apertium start
    and end of token, they are regex start and end
    of line. Token is assumed to have been already
    stripped of its ^ and $.
    """

    # start with the lemma (or with the lack of it)
    re_line = '^' + cat_item.attrib.get('lemma', '[^<>]*')

    tags = cat_item.attrib['tags']

    if tags == '':
        # no tags: close regex line
        return re_line + '$'

    tag_sequence = tags.split('.')
    for tag in tag_sequence[:-1]:
        if tag == '*':
            # any tag
            re_line += any_tag_re
        else:
            # specific tag
            re_line += '<{}>'.format(tag)

    if tag_sequence[-1] == '*':
        # any tags at the end
        re_line += any_num_of_any_tags_re
    else:
        # specific tag at the end
        re_line += '<{}>'.format(tag_sequence[-1])

    return re_line + '$'


def get_cat_dict(transtree):
    """
    Get an xml tree with transfer rules.
    Build an inverted index of the rules.
    """
    root = transtree.getroot()
    cat_dict = {}
    for def_cat in root.find('section-def-cats').findall('def-cat'):
        # make a regex line to recognize lemma-tag pattern
        re_line = '|'.join(cat_item_to_re(cat_item)
                            for cat_item in def_cat.findall('cat-item'))
        # add empty category list if there is none
        cat_dict.setdefault(re_line, [])
        # add category to the list
        cat_dict[re_line].append(def_cat.attrib['n'])
    return cat_dict


def get_cats_by_line(line, cat_dict):
    """
    Return all possible categories for each apertium token in line.
    """

    return [get_cat(token, cat_dict)
                for token in apertium_token_re.findall(line)]


def get_cat(token, cat_dict):
    """
    Return all possible categories for token.
    """

    token_cat_list = []
    for cat_re, cat_list in cat_dict.items():
        if re.match(cat_re, token):
            token_cat_list.extend(cat_list)
    return (token, token_cat_list)


def get_rules(transtree):
    """
    From xml tree with transfer rules,
    get rules, ambiguous rules,
    and rule id to number map.
    """
    root = transtree.getroot()

    # build pattern -> rules numbers dict (rules_dict),
    # and rule number -> rule id dict (rule_id_map)
    rules_dict, rule_xmls, rule_id_map  = {}, {}, {}
    for i, rule in enumerate(root.find('section-rules').findall('rule')):
        if 'id' in rule.attrib:
            # rule has 'id' attribute: add it to rule_id_map
            rule_id_map[str(i)] = rule.attrib['id']
            rule_xmls[str(i)] = rule
        # build pattern
        pattern = tuple(pattern_item.attrib['n'] 
                for pattern_item in rule.find('pattern').findall('pattern-item'))
        # add empty rules list for pattern
        # if pattern was not in rules_dict
        rules_dict.setdefault(pattern, [])
        # add rule number to rules list
        rules_dict[pattern].append(str(i))

    # detect groups of ambiguous rules,
    # and prepare rules for building FST
    rules, ambiguous_rule_groups = [], {}
    for pattern, rule_group in rules_dict.items():
        if all(rule in rule_id_map for rule in rule_group):
            # all rules in group have ids: add group to ambiguous rules
            ambiguous_rule_groups[rule_group[0]] = rule_group
        # add pattern to rules using first rule as default
        rules.append(pattern + (rule_group[0],))
    # sort rules to optimize FST building
    rules.sort()

    return rules, ambiguous_rule_groups, rule_id_map, rule_xmls


def prepare(rfname):
    """
    Read transfer file and prepare pattern FST.
    """
    try:
        transtree = ET.parse(rfname)
    except FileNotFoundError:
        print('Failed to locate rules file \'{}\'. '
              'Have you misspelled the name?'.format(opts.rfname))
        sys.exit(1)
    except ET.ParseError:
        print('Error parsing rules file \'{}\'. '
              'Is there something wrong with it?'.format(opts.rfname))
        sys.exit(1)


    cat_dict = get_cat_dict(transtree)
    rules, ambiguous_rules, rule_id_map, rule_xmls = get_rules(transtree)

    return cat_dict, rules, ambiguous_rules, rule_id_map, rule_xmls


class FST:
    """
    FST for coverage recognition.
    """
    def __init__(self, init_rules):
        """
        Initialize with patterns from init_rules.
        """
        self.start_state = 0
        self.final_states = {} # final state: rule
        self.transitions = {} # (state, input): state

        maxlen = max(len(rule) for rule in init_rules)
        self.maxlen = maxlen - 1

        
        # make rule table, where each pattern starts with ('start', 0)
        rules = [[('start', self.start_state)] + list(rule) for rule in init_rules]

        state, prev_cat = self.start_state, ''
        # look at each rule pattern at fixed position 
        for level in range(1, maxlen):
            for rule in rules:
                if len(rule) <= level:
                    # this rule already ended: increment state to keep it simple
                    state += 1
                elif len(rule) == level+1:
                    # end of the rule is here: add this state as a final
                    self.final_states[rule[level-1][1]] = rule[level]
                else:
                    if rule[level] != prev_cat:
                        # rule patterns diverged: add new state                        
                        state += 1
                    # add transition
                    self.transitions[(rule[level-1][1], rule[level])] = state
                    prev_cat = rule[level]
                    # add current state to current pattern element
                    rule[level] = (rule[level], state)
            # change prev_cat to empty at the end of rules list
            # to ensure state is changed at the start of next run through
            prev_cat = ''


    def get_lrlm(self, line, cat_dict):
        """
        Build all lrlm coverages for line.
        
        """
        # tokenize line and get all possible categories for each token
        line = get_cats_by_line(line, cat_dict)
        # coverage and state lists are built dinamically
        # each state from state_list is the state of FST
        # at the end of corresponding coverage from coverage_list
        coverage_list, state_list = [[]], [self.start_state]

        for token, cat_list in line:
            new_coverage_list, new_state_list = [], []

            if cat_list == []:
                for coverage, state in zip(coverage_list, state_list):
                    if state in self.final_states:
                        new_coverage = coverage + [('r', self.final_states[state])]
                    else:
                        new_coverage = coverage + [('r', 'default')] 

                    new_coverage_list.append(new_coverage + [('w', token), ('r', 'default')])
                    new_state_list.append(self.start_state)
            else:
                # go through all cats for the token
                for cat in cat_list:
                    # try to continue each coverage obtained on the previous step
                    for coverage, state in zip(coverage_list, state_list):
                        # first, check if we can go further along current pattern
                        if (state, cat) in self.transitions:
                            # current pattern can be made longer: add one more token
                            new_coverage_list.append(coverage + [('w', token)])
                            new_state_list.append(self.transitions[(state, cat)])

                        # if not, check if we can finalize current pattern
                        elif state in self.final_states:
                            # current state is one of the final states: close previous pattern
                            new_coverage = coverage + [('r', self.final_states[state])]

                            if (self.start_state, cat) in self.transitions:
                                # can start new pattern
                                new_coverage_list.append(new_coverage + [('w', token)])
                                new_state_list.append(self.transitions[(self.start_state, cat)])
                            elif '*' in token:
                                # can not start new pattern because of an unknown word
                                new_coverage_list.append(new_coverage + [('w', token), ('r', 'unknown')])
                                new_state_list.append(self.start_state)
                            else:
                                new_coverage_list.append(new_coverage + [('w', token), ('r', 'default')])
                                new_state_list.append(self.start_state)                             

                        # if not, check if it is just an unknown word
                        elif state == self.start_state and '*' in token:
                            # unknown word at start state: add it to pattern, start new
                            new_coverage_list.append(coverage + [('w', token), ('r', 'unknown')])
                            new_state_list.append(self.start_state)

                        else:
                            if coverage == [] or coverage[-1][0] == 'r':
                                new_coverage_list.append(coverage + [('w', token), ('r', 'default')])
                                new_state_list.append(self.start_state)
                            else:
                                try:
                                    new_state_list.append(self.transitions[(self.start_state, cat)]) 
                                    new_coverage_list.append(coverage + [('r', 'default'), ('w', token)])
                                except:
                                    new_state_list.append(self.start_state) 
                                    new_coverage_list.append(coverage + [('r', 'default'), ('w', token), ('r', 'default')])
           
            def_coverage_list, def_state_list = [], []
            cleaned_coverage_list, cleaned_state_list = [], []
            
            for coverage, state in zip(new_coverage_list, new_state_list):
                try:
                    if coverage[-1][0] == 'r' and coverage[-1][-1] == 'default':
                        def_coverage_list.append(coverage)
                        def_state_list.append(state)
                    elif len(coverage) > 1 and coverage[-2][0] == 'r' and coverage[-2][-1] == 'default':
                            def_coverage_list.append(coverage)
                            def_state_list.append(state)
                    else:
                        cleaned_coverage_list.append(coverage)
                        cleaned_state_list.append(state)
                except:
                    continue

            if len(cleaned_coverage_list) == 0:
                coverage_list, state_list = def_coverage_list, def_state_list
            else:
                coverage_list, state_list = cleaned_coverage_list, cleaned_state_list          

        # finalize coverages
        new_coverage_list = []
        for coverage, state in zip(coverage_list, state_list):
            if state in self.final_states:
                # current state is one of the final states: close the last pattern
                new_coverage_list.append(coverage + [('r', self.final_states[state])])
            elif coverage != [] and coverage[-1][0] == 'r':
                # the last pattern is already closed
                new_coverage_list.append(coverage)
            # if nothing worked, just discard this coverage as incomplete

        if new_coverage_list == []:
            # no coverages detected: no need to go further
            return []

        # convert coverage representation:
        # [('r'/'w', rule_number/token), ...] -> [([token, token, ... ], rule_number), ...]
        formatted_coverage_list = []
        for coverage in new_coverage_list:
            pattern, formatted_coverage = [], []
            for element in coverage:
                if element[0] == 'w':
                    pattern.append(element[1])
                else:
                    formatted_coverage.append((pattern, element[1]))
                    pattern = []
            formatted_coverage_list.append(formatted_coverage)
        return formatted_coverage_list


def return_output_from_shell(command):
    """Find and return location of apertium-lang-pair directory"""

    shell_output = check_output(command, shell=True)
    shell_output= str(shell_output).split('\\n')
    shell_output = shell_output[0][2:]

    return shell_output


def preprocess_corpus(corpus):
    with open(corpus, 'r', encoding='utf-8') as file:
        corpus = file.readlines()

    corpus = ''.join(corpus)

    corpus = re.sub('\' s', '\'s', corpus)
    corpus = re.sub('\(', '', corpus)
    corpus = re.sub('\)', '', corpus)
    corpus = re.sub('""', '"', corpus)
    corpus = corpus.split('\n')

    return corpus


def tag_corpus(corpus, lang_pair):
    tagged_corpus = []
    wrong_sentences = 0

    for sentence in corpus:
        try:
            command = 'echo "' + sentence + '" | apertium -d . ' + lang_pair + '-pretransfer'
            tagged_sentence = return_output_from_shell(command)           
            tagged_corpus.append(tagged_sentence)
        except:
            wrong_sentences += 1

    print('Number of incorrect sentences: %s' % (wrong_sentences))

    return tagged_corpus


def calculate_coverages(tagged_corpus, rules, cat_dict):
    cov_lengths, ij_errors, guio_errors, other_errors = [], [], [], []
    errors_counter = 0
    pattern_FST = FST(rules)

    for sentence in tagged_corpus[:-1]:
        coverages = pattern_FST.get_lrlm(sentence, cat_dict)
        current_length = len(coverages)

        print(current_length)

        if current_length != 0:
            cov_lengths.append(current_length)
        else:
            errors_counter += 1


    mean_length = sum(cov_lengths) / len(cov_lengths)
    errors_percent = errors_counter * 100 / len(tagged_corpus)
   
    print('Mean number of coverages: %s' % (mean_length))
    print('Total percentage of errors: %s' % (errors_percent))


def main():
    corpus = sys.argv[1]
    lang_pair = sys.argv[2]
    path_to_lang_pair = sys.argv[3]
    t1x_file = 'apertium-' + lang_pair + '.' + lang_pair + '.t1x'
    
    os.chdir(path_to_lang_pair)

    preprocessed_corpus = preprocess_corpus(corpus)
    tagged_corpus = tag_corpus(preprocessed_corpus, lang_pair)
    cat_dict, rules, ambiguous_rules, rule_id_map, rule_xmls = prepare(t1x_file)
    calculate_coverages(tagged_corpus, rules, cat_dict)


if __name__ == '__main__':
    main()
