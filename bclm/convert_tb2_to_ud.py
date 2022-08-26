"""
 Writers:
 - Eylon Gueta for most of the code
 - Refael Shaked Greenfeld on some fixes and enhancements
"""

import argparse

from conllu import parse_incr
import itertools
from conllu import serializer
import conllu


def add_morph(tl, morph, i, append_multi_morph=False):
    # id
    if i == len(tl):
        next_id = tl[-1]['id'] + 1
    else:
        next_id = tl[i]['id']
        if isinstance(next_id, tuple):
            next_id = next_id[0]

    morph['id'] = next_id
    morph = copy_token(morph)

    # extend existing multi-morph
    prev_multi_morph_token_i = last_index(tl[:i], lambda j, d: isinstance(d['id'], tuple))
    if prev_multi_morph_token_i != -1:
        d = tl[prev_multi_morph_token_i]
        start, _, end = d['id']
        if start <= next_id and (next_id <= end or (append_multi_morph and next_id == end + 1)):
            d['id'] = (start, d['id'][1], end + 1)

    # offset ids
    for d in tl[i:]:
        if isinstance(d['id'], tuple):
            start, _, end = d['id']
            d['id'] = (start + 1, d['id'][1], end + 1)
        else:
            d['id'] += 1
    # fix heads by increasing index by 1
    for d in tl:
        if 'head' in d and d['head'] and d['head'] >= tl[i]["id"] - 1:
            d['head'] += 1

    # new TokenList
    morph['head'] = morph['id'] + 1
    tl_new = conllu.models.TokenList(tl[:i] + [morph] + tl[i:], metadata=tl.metadata)

    return tl_new


def read_conllu(data_path, max_rows=None):
    data = []

    with open(data_path) as f:
        g = parse_incr(f)
        if max_rows is not None:
            g = itertools.islice(g, max_rows)

        for tl in g:
            data.append(tl)

    return data


def get_feat_k(d, k):
    if 'feats' in d and isinstance(d['feats'], dict):
        return d['feats'].get(k, None)

    return None


def is_feat_k_v(d, k, v):
    return 'feats' in d and isinstance(d['feats'], dict) and k in d['feats'] and d['feats'][k] == v


def make_create_morph(morph):
    def _f():
        return conllu.models.Token(morph)

    return _f


fl_morph = conllu.models.Token(
    {'form': 'של', 'lemma': 'של', 'upos': 'ADP', 'xpos': 'ADP', 'feats': {'Case': 'Gen'}, 'deprel': 'case:gen',
     'deps': None, 'misc': None, })

at_morph = conllu.models.Token(
    {'form': 'את', 'lemma': 'את', 'upos': 'ADP', 'xpos': 'ADP', 'feats': {'Case': 'Acc'}, 'deprel': 'case:acc',
     'deps': None, 'misc': None, })

h_det_morph = conllu.models.Token(
    {'form': 'ה_', 'lemma': 'ה', 'upos': 'DET', 'xpos': 'DET', 'feats': {'PronType': 'Art', 'Definite': 'Def'},
     'deprel': 'det', 'deps': None, 'misc': None, })

create_fl_morph = make_create_morph(fl_morph)
create_at_morph = make_create_morph(at_morph)
create_h_det_morph = make_create_morph(h_det_morph)

def copy_token(t):
    t2 = {**t}
    if 'feats' in t2 and isinstance(t2['feats'], dict):
        t2['feats'] = {**t2['feats']}

    return conllu.models.Token(t2)


def copy_tl(tl):
    return conllu.models.TokenList([copy_token(t) for t in tl], metadata=tl.metadata)


def last_index(last_idx, func):
    for j in range(len(last_idx) - 1, -1, -1):
        d = last_idx[j]
        if func(j, d):
            return j
    return -1


def dict_submatch_dict(d_query, d_value):
    if any((qk not in d_value for qk in d_query)):
        return False

    for k, v in d_query.items():
        vv = d_value[k]

        # function
        if isinstance(v, type(lambda: None)):
            if not v(vv):
                return False
        # dict
        elif isinstance(v, dict) and isinstance(vv, dict):
            if not dict_submatch_dict(v, vv):
                return False
        elif v != vv:
            return False

    return True


def search_iter(data, morphs_pattern):
    len_pattern = len(morphs_pattern)

    for i, tl in enumerate(data):
        js = []
        for j in range(len(tl) - len_pattern):
            morphs = tl[j: j + len_pattern]

            if all(dict_submatch_dict(mq, mv) for mq, mv in zip(morphs_pattern, morphs)):
                js.append(j)
        if js:
            yield i, js


feats_to_person_morph = {(None, 'Sing', '1'): 'אני', ('Fem', 'Sing', '1'): 'אני', ('Masc', 'Sing', '1'): 'אני',
                         ('Fem,Masc', 'Sing', '1'): 'אני',

                         (None, 'Plur', '1'): 'אנחנו', ('Fem,Masc', 'Plur', '1'): 'אנחנו',
                         ('Masc', 'Plur', '1'): 'אנחנו', ('Fem', 'Plur', '1'): 'אנחנו',

                         ('Masc', 'Sing', '2'): 'אתה', ('Masc', 'Sing', '3'): 'הוא', ('Masc', 'Plur', '2'): 'אתם',
                         ('Masc', 'Plur', '3'): 'הם', ('Fem', 'Sing', '2'): 'את', ('Fem', 'Sing', '3'): 'היא',
                         ('Fem', 'Plur', '2'): 'אתן', ('Fem', 'Plur', '3'): 'הן', }


def get_person_form(d):
    gen = get_feat_k(d, 'Gender')
    num = get_feat_k(d, 'Number')
    per = get_feat_k(d, 'Person')
    k = (gen, num, per)

    if k not in feats_to_person_morph:
        return d['form']

    return feats_to_person_morph[k]


per_forms = set(feats_to_person_morph.values())

morphs_pattern_per = [{'upos': lambda x: x != 'PUNCT'}, {'lemma': 'הוא', 'form': lambda x: x not in per_forms}]

# ### covert det h

# In[15]:


morph_pattern_covert_det = [{'form': lambda x: x in {'ל', 'ב', 'כ'}, 'lemma': lambda x: x != 'ה',
                             'feats': {'Definite': 'Def', 'PronType': 'Art', }}]


def ud_preprocess(data):
    for tl in data:
        for i, d in enumerate(tl):
            f = d['form']

            if len(f) > 1 and '_' in f:
                d['form'] = f.replace('_', '')
            if d['deprel'] == "compound":
                d['deprel'] = "compound:smixut"
            if d['deprel'] == "flat":
                d['deprel'] = "flat:name"


def tb2_preprocess(data):
    # מילה + (של/את) + הוא
    for i, js in search_iter(data, morphs_pattern_per):
        tl = data[i]

        for j in sorted(js, reverse=True):
            try:
                d_host = tl[j]
                d_per = tl[j + 1]

                # e.g. ממנ + ו -> מן + הוא
                # e.g. עבודת + ו -> עבודה + הוא
                d_host['form'] = d_host['lemma']
                d_per['form'] = get_person_form(d_per)

                # add של/את
                morph_case = get_feat_k(d_per, 'Case')
                if morph_case in ('Acc', 'Gen'):
                    if morph_case == 'Acc':
                        morph = create_at_morph()
                    else:
                        morph = create_fl_morph()

                    d_per['feats'].pop('Case')

                    tl = add_morph(tl, morph, j + 1)
            except:
                print(i, j)
                raise

        data[i] = tl

    # covert ה
    for i, js in search_iter(data, morph_pattern_covert_det):
        tl = data[i]

        for j in sorted(js, reverse=True):
            d = tl[j]
            d['feats'].pop('Definite')
            d['feats'].pop('PronType')
            if len(d['feats']) == 0:
                d['feats'] = None

            morph = create_h_det_morph()
            tl = add_morph(tl, morph, j + 1)

        data[i] = tl


def reorder_dict(d, keys):
    return {k: d.get(k, None) for k in keys}


def serialize_tl(tl, tl_fields_order=conllu.parser.DEFAULT_FIELDS):
    tl_ordered = conllu.models.TokenList([reorder_dict(t, tl_fields_order) for t in tl], metadata=tl.metadata)

    return serializer.serialize(tl_ordered)


def convert_to_htb2(input_path):
    ud_data = read_conllu(input_path)
    ud_preprocess(ud_data)
    tb2_preprocess(ud_data)
    return ud_data


def parse_arguments():
    p = argparse.ArgumentParser(description='Convert new htb format to old format')
    p.add_argument('input', help="input file expect UD conll file")
    p.add_argument('output', help="output file chunked to NP")
    p.add_argument('-a', '--all', dest='all', action='store_true', help="support {spacy, trankit} parser")
    return p.parse_args()


def main():
    args = parse_arguments()
    ud_data = convert_to_htb2(args.input)
    with open(args.output, 'w') as f:
        for tl in ud_data:
            f.write(serialize_tl(tl))


if __name__ == '__main__':
    main()


