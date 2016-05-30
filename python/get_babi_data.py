import re
from itertools import groupby

DATAPATH = '../data/babi/tasks_1-20_v1-2/en-10k/'
# Functions to load babi dataset

def get_task_name(task_num):
    if task_num == 1:
        return 'qa1_single-supporting-fact'
    elif task_num == 2:
        return 'qa2_two-supporting-facts'
    elif task_num == 3:
        return 'qa3_three-supporting-facts'
    elif task_num == 4:
        return 'qa4_two-arg-relations'
    elif task_num == 5:
        return 'qa5_three-arg-relations'
    elif task_num == 6:
        return 'qa6_yes-no-questions'
    elif task_num == 7:
        return 'qa7_counting'
    elif task_num == 8:
        return 'qa8_lists-sets'
    elif task_num == 9:
        return 'qa9_simple-negation'
    elif task_num == 10:
        return 'qa10_indefinite-knowledge'
    elif task_num == 11:
        return 'qa11_basic-coreference'
    elif task_num == 12:
        return 'qa12_conjunction'
    elif task_num == 13:
        return 'qa13_compound-coreference'
    elif task_num == 14:
        return 'qa14_time-reasoning'
    elif task_num == 15:
        return 'qa15_basic-deduction'
    elif task_num == 16:
        return 'qa16_basic-induction'
    elif task_num == 17:
        return 'qa17_positional-reasoning'
    elif task_num == 18:
        return 'qa18_size-reasoning'
    elif task_num == 19:
        return 'qa19_path-finding'
    else:
        return 'qa20_agents-motivations'

def get_task_train(task_num):
    task_name = get_task_name(task_num)
    return get_babi_dataset(DATAPATH + task_name + '_train.txt')

def get_task_test(task_num):
    task_name = get_task_name(task_num)
    return get_babi_dataset(DATAPATH + task_name + '_test.txt')

# Returns a list of (input, question, answer) tuples for the training set
def get_task_6_train():
    return get_babi_dataset('../data/babi/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_train.txt')
    #return get_babi_dataset('../data/babi/tasks_1-20_v1-2/qa6_tiny_train.txt')

# Returns a list of (input, question, answer) tuples for the training set
def get_task_6_test():
    return get_babi_dataset('../data/babi/tasks_1-20_v1-2/en-10k/qa6_yes-no-questions_test.txt')
    #return get_babi_dataset('../data/babi/tasks_1-20_v1-2/qa6_tiny_test.txt')

# Returns a list of (input, question, answer) tuples
def get_babi_dataset(path):
    print "Loading babi data"
    babi_data_path = path

    babi_data = open(babi_data_path).read()
    lines = data_to_list(babi_data)

    data, story = [], []
    for line in lines:
        nid, line = line.strip().split(' ', 1)

        # Check for start of a new story
        if int(nid) == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            substory = [x for x in story if x]
            data.append((substory, tokenize(q), a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)

    return [(flatten(_story), _question, answer) for _story, _question, answer in data]

def remove_long_sentences(data, max_len):
    clean_data = []

    for vec in data:
        sentence_vec = [list(group) for k, group in groupby(vec[0], lambda x: x == ".") if not k]
        num_sentences = len(sentence_vec)

        if num_sentences > max_len:
            sentence_vec = sentence_vec[-max_len:]

        reconstructed_vec = []
        for sentence in sentence_vec:
            reconstructed_vec += sentence
            reconstructed_vec.append(unicode('.'))

        clean_data.append((reconstructed_vec, vec[1], vec[2]))

    return clean_data


def tokenize(sentence):
    """
    Split a sentence into tokens including punctuation.
    Args:
        sentence (string) : String of sentence to tokenize.
    Returns:
        list : List of tokens.
    """
    return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]


def flatten(data):
    """
    Flatten a list of data.
    Args:
        data (list) : List of list of words.
    Returns:
        list : A single flattened list of all words.
    """
    return reduce(lambda x, y: x + y, data)


def data_to_list(data):
    """
    Clean a block of data and split into lines.
    Args:
        data (string) : String of bAbI data.
    Returns:
        list : List of cleaned lines of bAbI data.
    """
    split_lines = data.split('\n')[:-1]
    return [line.decode('utf-8').strip() for line in split_lines]
