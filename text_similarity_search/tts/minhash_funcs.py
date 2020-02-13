import glob
# import spacy
import io
import os
# noinspection PyPep8Naming
import pickle
from collections import defaultdict
from hashlib import md5
from itertools import repeat
from multiprocessing import Pool
from random import shuffle

import numpy as np
import regex as re
from datasketch import MinHash, MinHashLSH, LeanMinHash
from tqdm import tqdm

# from spacy import en, language_data


N_PERM = 256  # Globally defined number of permutations for min-hash indices.


def get_line_number(file_path):
    """ Return total number of lines in file_path"""
    lines = 0
    for line in open(file_path, 'r'):
        lines += 1
    return lines


# nlp = spacy.load('en')
# nlp = spacy.load('en_core_web_md')


def train_test_split(yelp_file, out_prefix):
    """

    Args:
        yelp_file: Input file to split into 3 parts
        out_prefix: prefix for output train/test/valid split.

    Returns: None - outputs are written to out_prefix+ train / test / valid.txt

    """
    np.random.seed(0)
    proportions = [0.999, 0.0005, 0.0005]
    lnum = get_line_number(yelp_file)
    with io.open(yelp_file, 'r', encoding='utf-8', errors='ignore') as fopen, \
            io.open(out_prefix + '.train.txt', 'w', encoding='utf-8') as ftrain, \
            io.open(out_prefix + '.test.txt', 'w', encoding='utf-8') as ftest, \
            io.open(out_prefix + '.valid.txt', 'w', encoding='utf-8') as fvalid:
        out_dict = {0: ftrain, 1: ftest, 2: fvalid}
        for line in tqdm(fopen, total=lnum):
            npc = np.random.choice(3, 1, p=np.array(proportions))[0]
            out_handle = out_dict[npc]
            out_handle.write(line)


def byte_counter(filename, blocksize):
    """ Returns byte offsets for filename such that each byte offset contains blocksize lines"""
    byte_id = [0]
    l_acc = 0
    with io.open(filename, 'r', encoding='utf-8', errors='ignore') as fhandle:
        line = fhandle.readline()
        while line:
            if l_acc > blocksize:
                pos = fhandle.tell()
                byte_id.append(pos)
                l_acc = 0
            else:
                l_acc += 1
            line = fhandle.readline()
        # if l_acc > 0:
        #    pos = fhandle.tell()
        #    byte_id.append(pos)
    return byte_id


def lsh_partial_wrap(arg):
    """ Wrapper function which calls make_lsh_partial via the mulitprocessing library"""
    return make_lsh_partial(arg[0], arg[1], arg[2], arg[3], arg[4], thresh=arg[5])


def make_lsh_partial(batch_id, batch_size, filename, out_filename, byte_start, nperm=N_PERM, thresh=0.5):
    """
    Generate the LSH index over a subset of the data.
    :param batch_id: Batch id, used to determine output filename
    :param batch_size: Specifies number of lines of the file to read
    :param filename: Input file, generated using the make_lsh_file family of functions
    :param out_filename: Output file prefix, batch_id is appended to distinguish each block.
    :param byte_start: Byte offset for the partial file - this allows make_lsh_partial to read the middle sections of
    a file using the seek() command.
    :param nperm: number of permutations in the Min-Hash index.
    :param thresh: Jaccard index threshold to return
    :return: filename of the dumped LSH file.
    """
    lsh = MinHashLSH(threshold=thresh, num_perm=nperm)
    current_batch = 0
    with io.open(filename, 'r', encoding='utf-8', errors='ignore') as fhandle:
        fhandle.seek(byte_start)
        for line in fhandle:
            lsplit = line.split(':')
            if len(lsplit) > 1:
                lnum = lsplit[0]
                line_sub = lsplit[1]
                wordlist = line_sub.split(' ')
                if len(wordlist) > 3 and (not lsh.__contains__(line_sub)):
                    lsh.insert(lnum + ':' + line_sub, make_hash(wordlist, nperm))
                    # lsh.insert((lnum + ':' + line_sub).encode(encoding='utf8'), make_hash(wordlist, nperm))
            current_batch += 1
            if current_batch >= batch_size:
                break
    outfile = out_filename + '_' + str(batch_id) + '.obj'
    dump_lsh(lsh, outfile)
    return outfile


# noinspection PyArgumentList
def parallel_make_lsh(lem_dir, out_dir, nproc, batch_size=1000 * 100, thresh=0.5):
    """ Split the lemmatized corpus file and dispatch each chunk to a make_partial_lsh call to make a LSH index."""
    bytelen = byte_counter(lem_dir, batch_size)
    pool = Pool(processes=nproc)
    file_list_out = []
    for fout in tqdm(pool.imap_unordered(
            lsh_partial_wrap,
            zip(range(len(bytelen)), repeat(batch_size), repeat(lem_dir), repeat(out_dir), bytelen, repeat(thresh))
    ),
            total=len(bytelen)):
        file_list_out.append(fout)
    pool.terminate()
    return file_list_out


def make_lsh(filename, out_filename, nperm=N_PERM, thresh=0.5, blocksize=1000000):
    """ Non-parallel variant of LSH caller - deprecated."""
    lsh = MinHashLSH(threshold=thresh, num_perm=nperm)
    batch_id = 0
    lsh_filenames = []
    current_batch = 0
    with io.open(filename, 'r', encoding='utf-8', errors='ignore') as fhandle:
        for line in tqdm(fhandle, total=get_line_number(filename)):
            lsplit = line.split(':')
            if len(lsplit) > 1:
                lnum = int(lsplit[0])
                line_sub = lsplit[1]
                wordlist = line_sub.split(' ')
                if len(wordlist) > 3 and (not lsh.__contains__(line_sub)):
                    lsh.insert(str(lnum) + ':' + line_sub, make_hash(wordlist, nperm))
                    current_batch += 1
                if current_batch > blocksize:
                    outfile = out_filename + '_' + str(batch_id) + '.obj'
                    dump_lsh(lsh, outfile)
                    lsh_filenames.append(outfile)
                    lsh = MinHashLSH(threshold=thresh, num_perm=nperm)
                    batch_id += 1
                    current_batch = 0
    if current_batch > 0:
        outfile = out_filename + '_' + str(batch_id) + '.obj'
        dump_lsh(lsh, outfile)
        lsh_filenames.append(outfile)
    return lsh_filenames


def jaccard(s1, s2):
    """ Exact jaccard index calculation for two sentences represented as sets of words"""
    return float(len(s1.intersection(s2))) / float(len(s1.union(s2)))


def lsh_wrapper(args):
    """ Wrapper function for querying the LSH index. Given a particular LSH index on disk (filename) and queries
    as a list of sets of words (setslist), returns a dictionary mapping each sentence query to its LSH matches
    """
    filename = args[0]
    setslist = args[1]
    sentences = args[2]
    jac_tr = args[3]
    query_results = defaultdict(list)
    lsh = load_lsh(filename)

    try:
        assert isinstance(lsh, MinHashLSH)
    except AssertionError:
        print('Unpickled data structure does not match LSH:')
        print(filename)
        print(type(lsh))
        return query_results
    try:
        for i in range(len(setslist)):
            candidate_sents = lsh.query(make_hash(setslist[i]))
            s1 = set(setslist[i])
            tmp_set = []
            for c_sent in candidate_sents:
                ssplit = c_sent.rstrip().split(':')[1].split(' ')
                s2 = set(ssplit)
                if jaccard(s1, s2) > jac_tr:
                    tmp_set.append(c_sent)
            query_results[sentences[i]] += tmp_set
    except:
        print('Exception occured with file')
        print(filename)
    return query_results


def lsh_query_parallel(lsh_filenames, setslist, sentences, nproc=1, jac_tr=0.5):
    """ Uses the LSH index on lsh_filenames to find all neighbors of sentences within jac_tr jaccard distance."""
    query_results = defaultdict(list)
    # noinspection PyArgumentList
    pool = Pool(processes=nproc)
    print('enter parallel part')
    for query_tmp in tqdm(pool.imap_unordered(lsh_wrapper,
                                              zip(lsh_filenames, repeat(setslist), repeat(sentences), repeat(jac_tr))),
                          smoothing=0.0, total=len(lsh_filenames)):
        for (k, v) in query_tmp.items():
            query_results[k].extend(v)
    print('exit parallel part')
    pool.terminate()
    return query_results


def lsh_query(lsh_filenames, setslist, sentences, jac_tr=0.5):
    """ Single-core version of lsh_query_parallel, see above documentation."""
    query_results = defaultdict(list)
    for filename in tqdm(lsh_filenames):
        tmp_dict = lsh_wrapper([filename, setslist, sentences, jac_tr])
        for (k, v) in tmp_dict.items():
            query_results[k].extend(v)
    return query_results


global_perm = MinHash(num_perm=N_PERM).permutations


def make_hash(string, nperm=N_PERM):
    """ Generates the hash for any string represented as a list of words. MD5 hash used for speed,
    and global_perm is globally defined to avoid repeatedly calling the random number generator"""
    mh = MinHash(num_perm=nperm, permutations=global_perm, hashobj=md5)
    for word in string:
        mh.update(word.encode(encoding='utf-8'))
    return LeanMinHash(mh)


def dump_lsh(lsh, filename):
    """ Serializes the LSH index (lsh) to filename using cPickle"""
    with open(filename, 'wb') as pickle_file:
        pickle.dump(lsh, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


def load_lsh(filename):
    """ Deserializes the LSH index"""
    with open(filename, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def grab_seeds(filename, num_lines):
    """ Grabs num_lines random lines from filename """
    sentence_list = []
    set_list = []
    np.random.seed(0)
    with io.open(filename, 'r', encoding='utf-8', errors='ignore') as fhandle:
        ltot = get_line_number(filename)
        line_ids = np.sort(np.random.choice(
            ltot, size=num_lines, replace=False))
        lskip = line_ids[0]
        num_sampled = 0
        ldiff = np.diff(line_ids)
        for line in tqdm(fhandle, total=ltot):
            if lskip <= 0:
                line_sub = line.split(':')[1]
                wordlist = line_sub.split(' ')
                if len(wordlist) > 1:
                    str_add = line.rstrip()
                    sentence_list.append(str_add)
                    set_list.append(wordlist)
                if num_sampled >= len(ldiff):
                    break
                lskip = ldiff[num_sampled] - 1
                num_sampled += 1
            else:
                lskip -= 1
    return set_list, sentence_list


def find_sent_id(sent):
    splts = sent.split(':')
    return splts[0]


def generate_next_set(queue_tmp, batch_neighbors, already_seen_set, adj_file):
    """
    Given a set of input sentences, dump an adjacency list and return the list of sentences to expand next.
    :param queue_tmp: the set of seed vertices to process
    :param batch_neighbors:  dictionary mapping from seed vertices to its neighbors
    :param already_seen_set: set (modified in place) of seeds already observed (so that we can skip in the future)
    :param adj_file: file handle for where to dump the outputs.
    :return: new seeds (ie neighbors which are not already explored seeds)
    """
    next_seed_sent = []
    for sent in tqdm(queue_tmp):
        nb_cands = batch_neighbors[sent]
        if len(nb_cands) > 1:
            # nbstring = sent.rstrip() + u',' \
            #            + u','.join([strsub.decode('utf-8').rstrip() for strsub in nb_cands]) \
            #            + u'\n'

            nbstring = find_sent_id(sent.rstrip()) + ',' \
                       + ','.join([find_sent_id(strsub.rstrip()) for strsub in nb_cands]) \
                       + '\n'

            bytes_written = adj_file.write(nbstring)
            # bytes_written = adj_file.write(nbstring2)

            for cand in nb_cands:
                candsent = cand.rstrip().split(':')[1]
                if candsent not in already_seen_set:
                    next_seed_sent.append(cand.rstrip())
                    already_seen_set.add(candsent)
    return next_seed_sent


def generate_next_set_partial(args):
    batch_id = args[0]
    batch_size = args[1]
    batch_neighbors = args[2]
    already_seen_set = args[3]
    queue_tmp = args[4]
    adjlist_prefix = args[5]

    queue_tmp = queue_tmp[batch_id * batch_size: (batch_id + 1) * batch_size]

    adj_file = io.open(adjlist_prefix + '_p_' + str(batch_id), 'w', encoding='utf8')

    next_seed_sent = []
    for sent in tqdm(queue_tmp):
        nb_cands = batch_neighbors[sent]
        if len(nb_cands) > 1:
            nbstring = sent.rstrip() + ',' + ','.join(
                [strsub.rstrip() for strsub in nb_cands]) + '\n'
            bytes_written = adj_file.write(nbstring)
            for cand in nb_cands:
                candsent = cand.rstrip().split(':')[1]
                if candsent not in already_seen_set:
                    next_seed_sent.append(cand.rstrip())
                    # already_seen_set.add(candsent)

    adj_file.close()

    return next_seed_sent


def generate_next_set_parallel(queue_tmp, batch_neighbors, already_seen_set, adj_file, nproc, adjlist_prefix):
    batch_size = len(queue_tmp) // nproc

    alss = already_seen_set
    next_seed_sent = []
    pool = Pool(processes=nproc)
    for result in pool.imap_unordered(
            generate_next_set_partial,
            zip(range(nproc), repeat(batch_size), repeat(batch_neighbors), repeat(alss), repeat(queue_tmp),
                repeat(adjlist_prefix))
    ):
        next_seed_sent.extend(result)
        print('done generating next set split')

    pool.terminate()

    next_seed_sent = set(next_seed_sent)
    already_seen_set.update(next_seed_sent)

    splits = glob.glob(adjlist_prefix + '_p_*')
    for p in splits:
        with io.open(p, 'r', encoding='utf8') as f:
            for l in f:
                adj_file.write(l[:-1] + '\n')

        os.remove(p)

    return list(next_seed_sent)


def generate_adjlist(queue_input, batch_size, prefix, lsh_filename, n_rounds=10, nproc=1, jac_tr=0.5):
    """
    Generate an adjacency_list formatted graph by performing a breadth-first-search starting at queue_input, using
    graph connectivity implied by teh LSH index at lsh_filename.
    :param queue_input: Seed sentences from which to start the BFS
    :param batch_size: Maximum number of sentences to expand at once in the BFS step. Larger is faster, but uses memory
    :param prefix: Ourput filename prefix. Dumps prefix+_adjlist.txt
    :param lsh_filename: List of filenames defining the LSH index.
    :param n_rounds: number of rounds of BFS to perform
    :param nproc:  Max number of processors to use, larger is faster but uses more memory
    :param jac_tr: Jaccard index cutoff
    :return: None, outputs are dumped to file.
    """
    adjacency_list_file = prefix + '_adjlist.txt'
    adj_file = io.open(adjacency_list_file, 'w',
                       encoding='utf-8', errors='ignore')
    #
    queue_tmp = queue_input[0:batch_size]
    queue_input = queue_input[batch_size:]
    already_seen_set = set([sent.split(':')[1] for sent in queue_tmp])
    for batch_id in range(n_rounds):
        print(batch_id)
        #
        batch_set_input = [input_sent.split(':')[1].split(
            ' ') for input_sent in queue_tmp]
        if nproc == 1:
            batch_neighbors = lsh_query(
                lsh_filename, batch_set_input, queue_tmp, jac_tr)
        else:
            batch_neighbors = lsh_query_parallel(
                lsh_filename, batch_set_input, queue_tmp, nproc, jac_tr)
        #
        print('generating next set')
        g1 = generate_next_set(queue_tmp, batch_neighbors, already_seen_set, adj_file)
        # g2 = generate_next_set_parallel(queue_tmp,batch_neighbors, already_seen_set, adj_file, nproc, adjacency_list_file)

        queue_input += g1
        shuffle(queue_input)
        adj_file.flush()
        queue_tmp = queue_input[0:batch_size]
        queue_input = queue_input[batch_size:]
        if len(queue_tmp) == 0:
            break
    adj_file.close()


def dedup_adjlist(prefix):
    """ Given an adajency list, removes any duplicate entries (ie neighbors which are identical)"""
    adjacency_list_file = prefix + '_adjlist.txt'
    adj_file = io.open(adjacency_list_file, 'r',
                       encoding='utf-8', errors='ignore')
    adj_file_out = io.open(prefix + '_adjlist_dedup.txt',
                           'w', encoding='utf-8')
    for sent in tqdm(adj_file, total=get_line_number(adjacency_list_file)):
        all_sents = sent.rstrip().split(',')
        core_sent = all_sents[0].split(':')[1]
        string_targets = [targets.split(':')[1] for targets in all_sents[1:]]
        string_deduplicated = list(set(string_targets))
        nbstring = core_sent + ',' + ','.join(string_deduplicated) + '\n'
        bytes_out = adj_file_out.write(nbstring)
    adj_file_out.close()
    adj_file.close()


def split_adjlist(orig_gpe_input, adjlist_input, prefix_out):
    """
    Maps the adjlist format into two parts: an integer only adjacency list for machine parsing, and a text version for
    human inspection
    :param orig_gpe_input: Non-lemmatized but named entitty recognized file to use as the human redable part
    :param adjlist_input: Adjacency list outputs from the LSH code
    :param prefix_out: Output prefix. Outputs will be _gpe.txt and _int.txt for text and _str_dict.obj for a dictionary
    which maps integer ids to named entity recognized text.
    :return: All outputs go to text.
    """
    sentence_map = dict()
    gpe_out = io.open(prefix_out + '_gpe.txt', 'w', encoding='utf-8')
    numadj_out = io.open(prefix_out + '_int.txt', 'w', encoding='utf-8')
    with io.open(orig_gpe_input, 'r', encoding='utf-8') as gpe_file:
        for line in tqdm(gpe_file, total=get_line_number(orig_gpe_input)):
            lsplit = line.split(':')
            sentence_map[int(lsplit[0])] = ':'.join(lsplit[1:])
    with io.open(adjlist_input, 'r', encoding='utf-8') as adjlist_file:
        for line in tqdm(adjlist_file, total=get_line_number(adjlist_input)):
            nblist_index = re.findall(r'[^|,][0-9]+\:', line)
            # nblist_index = re.findall(',[^|][0-9]+\:', line)
            nblist_ints = [neighbor.replace(':', '').replace(',', '') for neighbor in nblist_index]
            nblist_strings = [
                sentence_map[int(nb_id)].strip() for nb_id in nblist_ints]
            gpe_out.write('\n'.join(nblist_strings) + '\n\n')
            numadj_out.write(','.join(nblist_ints) + '\n')
    gpe_out.close()
    numadj_out.close()
    with open(prefix_out + '_str_dict.obj', 'wb') as pickle_file:
        pickle.dump(sentence_map, pickle_file,
                    protocol=pickle.HIGHEST_PROTOCOL)


def split_adjlist_partial(batch_id, batch_start, sentence_map, adjlist_input, prefix_out, batch_size):
    gpe_out = io.open(prefix_out + '_gpe_p_' + str(batch_id), 'w', encoding='utf-8')
    # numadj_out = io.open(prefix_out + '_int.txt', 'w', encoding='utf-8')
    with io.open(adjlist_input, 'r', encoding='utf-8') as adjlist_file:
        adjlist_file.seek(batch_start)
        num_lines = 0
        for line in tqdm(adjlist_file, total=get_line_number(adjlist_input)):
            nblist_index = re.findall(r'[^|,][0-9]+\:', line)
            # nblist_index = re.findall(',[^|][0-9]+\:', line)
            nblist_ints = [neighbor.replace(':', '').replace(',', '') for neighbor in nblist_index]
            nblist_strings = [sentence_map[int(nb_id)].strip() for nb_id in nblist_ints]
            gpe_out.write('\n'.join(nblist_strings) + '\n\n')
            # numadj_out.write(','.join(nblist_ints) + '\n')

            num_lines += 1
            if num_lines >= batch_size:
                break

    gpe_out.close()
    # numadj_out.close()


def split_adjlist_wrap(args):
    batch_id = args[0]
    batch_start = args[1]
    sentence_map = args[2]
    adjlist_input = args[3]
    prefix_out = args[4]
    batch_size = args[5]
    return split_adjlist_partial(batch_id, batch_start, sentence_map, adjlist_input, prefix_out, batch_size)


def split_adjlist_parallel(orig_gpe_input, adjlist_input, prefix_out, batch_size, nprco):
    sentence_map = dict()
    with io.open(orig_gpe_input, 'r', encoding='utf-8') as gpe_file:
        for line in tqdm(gpe_file, total=get_line_number(orig_gpe_input)):
            lsplit = line.split(':')
            sentence_map[int(lsplit[0])] = ':'.join(lsplit[1:])

    bytelen = byte_counter(adjlist_input, batch_size)
    pool = Pool(processes=nprco)
    for _ in pool.imap_unordered(
            split_adjlist_wrap,
            zip(range(len(bytelen)), bytelen, repeat(sentence_map), repeat(adjlist_input), repeat(prefix_out),
                repeat(batch_size))
    ):
        print("split_adjslist_part finished.")

    pool.terminate()

    gpe_out_part_files = glob.glob(prefix_out + '_gpe_p_*')
    gpe_out_part_files.sort()

    gpe_out = io.open(prefix_out + '_gpe.txt', 'w', encoding='utf-8')
    for part_file in gpe_out_part_files:
        with io.open(part_file, 'r', encoding='utf8') as f:
            for l in f:
                gpe_out.write(l[:-1] + '\n')

        os.remove(part_file)

    gpe_out.close()


def get_stringpart(str_):
    return ':'.join(str_.strip().split(':')[1:])


def deduplicate_lsh_files(out_prefix):
    already_seen = set()
    with io.open(out_prefix + '_lemmatized.txt', 'r') as lemfile, io.open(out_prefix + '_orig.txt', 'r') as ofile, \
            io.open(out_prefix + '_orig_gpe.txt', 'r') as ogfile, io.open(out_prefix + '_lemmatized_dedup.txt',
                                                                          'w') as dlemfile, \
            io.open(out_prefix + '_orig_dedup.txt', 'w') as dofile, io.open(out_prefix + '_orig_gpe_dedup.txt',
                                                                            'w') as dogfile:
        for line in ogfile:
            lem_line = lemfile.readline()
            orig_line = ofile.readline()
            assert line.split(':')[0] == lem_line.split(':')[0]
            #
            lstring = get_stringpart(line)
            if lstring not in already_seen:
                dlemfile.write(lem_line)
                dogfile.write(line)
                dofile.write(orig_line)
                already_seen.add(lstring)


def make_test_set(test_dir, edit_input, tot_out, maxbucket=10, split_vec=np.array([0.9, 0.05, 0.05]), lower_jac=0.5):
    """
    :param test_dir: output directory for test file
    :param edit_input: input adjacency list file with NER tags replaced (_adjlist_gpe.txt)
    :param tot_out: total number of example pairs to output
    :param split_vec: probability of train/test/valid split
    :return:
    """
    np.random.seed(0)
    train_output = test_dir + '/train.tsv'
    test_output = test_dir + '/test.tsv'
    valid_output = test_dir + '/valid.tsv'
    # dump_stop(test_dir)
    with io.open(edit_input, 'r', encoding='utf-8') as edit_in_file, io.open(train_output, 'w',
                                                                             encoding='utf-8') as train_out_file, \
            io.open(test_output, 'w', encoding='utf-8') as test_out_file, io.open(valid_output, 'w',
                                                                                  encoding='utf-8') as valid_out_file:
        handle_dict = {0: train_out_file, 1: test_out_file, 2: valid_out_file}
        total_dumped = 0
        with tqdm(total=tot_out) as pbar:
            while total_dumped < tot_out:
                cur_line = edit_in_file.readline()
                if len(cur_line.strip()) == 0:
                    break
                base = cur_line
                base_set = set(base.split(' '))
                current_size = 0
                while len(cur_line) > 1:
                    cur_line = edit_in_file.readline()
                    if (len(cur_line) > 1) and (current_size < maxbucket):
                        jv = jaccard(base_set, set(cur_line.split(' ')))
                        if (jv > lower_jac) and (jv < 1.0):
                            target = cur_line
                            npc = np.random.choice(3, 1, p=split_vec)[0]
                            _ = handle_dict[0].write(base.strip() + '\t' + target.strip() + '\n')
                            current_size += 1
                            total_dumped += 1
                pbar.update(current_size)


def make_test_set_int(test_dir, edit_input, tot_out, sentence_mapping, maxbucket=10, lower_jac=0.5):
    """
    :param test_dir: output directory for test file
    :param edit_input: input adjacency list file with NER tags replaced (_adjlist_gpe.txt)
    :param tot_out: total number of example pairs to output
    :param split_vec: probability of train/test/valid split
    :return:
    """
    np.random.seed(0)
    train_output = test_dir + '/train.tsv'
    # dump_stop(test_dir)

    with io.open(edit_input, 'r', encoding='utf-8') as edit_in_file, \
            io.open(train_output, 'w', encoding='utf-8') as train_out_file:
        total_dumped = 0
        with tqdm(total=tot_out) as pbar:
            while total_dumped < tot_out:
                cur_line = edit_in_file.readline()
                if len(cur_line.strip()) == 0:
                    break

                ids = cur_line.strip().split(',')
                base_id = ids[0]
                others_neighbor_ids = ids[1:1 + maxbucket]

                if base_id not in sentence_mapping:
                    continue
                base_set = set(sentence_mapping[base_id].split(' '))

                current_size = 0
                for n_id in others_neighbor_ids:
                    if not n_id in sentence_mapping:
                        continue
                    neighbor = sentence_mapping[n_id]
                    jv = jaccard(base_set, set(neighbor.split(' ')))
                    if (jv > lower_jac) and (jv < 1.0):
                        train_out_file.write(
                            sentence_mapping[base_id].strip() + '\t' + neighbor.strip() + '\n'
                        )
                        current_size += 1
                        total_dumped += 1

                pbar.update(current_size)


def make_ids_pairs(edit_input, tot_out, sentence_mapping, maxbucket=10, lower_jac=0.5):
    """
    :param test_dir: output directory for test file
    :param edit_input: input adjacency list file with NER tags replaced (_adjlist_gpe.txt)
    :param tot_out: total number of example pairs to output
    :param split_vec: probability of train/test/valid split
    :return:
    """
    np.random.seed(0)

    pairs = []
    with io.open(edit_input, 'r', encoding='utf-8') as edit_in_file:
        total_dumped = 0
        with tqdm(total=tot_out) as pbar:
            while total_dumped < tot_out:
                cur_line = edit_in_file.readline()
                if len(cur_line.strip()) == 0:
                    break

                ids = cur_line.strip().split(',')
                base_id = int(ids[0])
                others_neighbor_ids = list(map(lambda x: int(x), ids[1:1 + maxbucket]))

                if base_id >= len(sentence_mapping):
                    continue

                base_set = set(sentence_mapping[base_id].split(' '))

                current_size = 0
                for n_id in others_neighbor_ids:
                    if n_id >= len(sentence_mapping):
                        continue

                    neighbor = sentence_mapping[n_id]
                    jv = jaccard(base_set, set(neighbor.split(' ')))
                    if (jv > lower_jac) and (jv < 1.0):
                        pairs.append((base_id, n_id))
                        current_size += 1
                        total_dumped += 1

                pbar.update(current_size)

    return pairs


def make_zeroedit_test(test_dir, corpus, tot_out, split_vec=np.array([0.7, 0.2, 0.1])):
    """
    :param test_dir: output directory for test file
    :param corpus: corpus with NER tags and colon separated input (each line is of the form `100:SENTENCE SPACE SEPARATED')
    :param tot_out: total number of example pairs to output
    :param split_vec: probability of train/test/valid split
    :return:
    """
    np.random.seed(0)
    train_output = test_dir + '/train.tsv'
    test_output = test_dir + '/test.tsv'
    valid_output = test_dir + '/valid.tsv'
    # dump_stop(test_dir)
    with io.open(corpus, 'r', encoding='utf-8') as corpus_in_file, io.open(train_output, 'w',
                                                                           encoding='utf-8') as train_out_file, \
            io.open(test_output, 'w', encoding='utf-8') as test_out_file, io.open(valid_output, 'w',
                                                                                  encoding='utf-8') as valid_out_file:
        handle_dict = {0: train_out_file, 1: test_out_file, 2: valid_out_file}
        total_dumped = 0
        with tqdm(total=tot_out) as pbar:
            while total_dumped < tot_out:
                cur_line = corpus_in_file.readline()
                if len(cur_line.strip()) == 0:
                    break
                base = ':'.join(cur_line.strip().split(':')[1:])
                npc = np.random.choice(3, 1, p=split_vec)[0]
                _ = handle_dict[npc].write(base + '\t' + base + '\n')
                pbar.update(1)
                total_dumped += 1
