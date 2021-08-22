import pandas as pd
from conllu import parse
from collections import OrderedDict
import os
import numpy as np
from io import StringIO
from .local_paths import *


def read_dataframe(corpus, remove_duplicates=False, remove_very_similar=False, subset=None):
    df = pd.read_csv(DF_PATHS[corpus.lower()], low_memory=False)
    if subset is not None:
        df = df[df.set==subset]
    return df


def read_treebank_conllu(filepath_or_buffer, remove_duplicates=False, remove_very_similar=False,
                         expand_feats=True, expand_misc=True):
    # metadata must include sent_id (int)
    # if you want to remove duplicates or very similar, metadata must also include 
    # duplicate_sent_id and very_similar_sent_id
    if isinstance(filepath_or_buffer, str):
        with open(tokens_filepath_or_buffer, 'r', encoding='utf8') as f:
            sp_conllu = parse(f.read())
    elif isinstance(filepath_or_buffer, StringIO):
        sp_conllu = parse(filepath_or_buffer.read())
        
    fixed = []
    dup_to_remove = set()
    very_sim_to_remove = set()
    for tl in sp_conllu:
        if (remove_duplicates and int(tl.metadata['sent_id']) in dup_to_remove 
            or remove_very_similar and int(tl.metadata['sent_id']) in very_sim_to_remove):
            print ('skipped', tl.metadata['sent_id'])
            continue
        for tok in tl:
            t = OrderedDict(tok)
            if type(t['id']) is not tuple:
                if expand_feats:
                    if t['feats'] is not None:
                        t.update({'feats_'+f: v for f, v in t['feats'].items()})
                    del(t['feats'])
                if expand_misc:
                    if t['misc'] is not None:
                        t.update({f: v for f, v in t['misc'].items()})
                    del(t['misc'])
                t.update(tl.metadata)
                fixed.append(t)
            if remove_duplicates:
                dup_to_remove = dup_to_remove | set(eval(tl.metadata['duplicate_sent_id']))
            if remove_very_similar:
                very_sim_to_remove = dup_to_remove | set(eval(tl.metadata['very_similar_sent_id']))

    df = pd.DataFrame(fixed)
    #sent_id required
    df['sent_id'] = df.sent_id.astype(int)
    
    if global_sent_id in df.columns:
          df['global_sent_id'] = df.global_sent_id.astype(int)
    if token_id in df.columns:
          df['token_id'] = df.misc_token_id.astype(int)

    return df


def read_conll(filepath_or_buffer, add_head_stuff=False, comment='#'):
    # CoNLL file is tab delimeted with no quoting
    # quoting=3 is csv.QUOTE_NONE
    if isinstance(filepath_or_buffer, str):
        buffer = open(filepath_or_buffer, 'r', encoding='utf8')
    elif isinstance(filepath_or_buffer, StringIO):
        buffer = filepath_or_buffer
        
    df = (pd.read_csv(buffer, sep='\t', header=None, quoting=3, comment=comment,
                names = ['id', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'head', 'deprel', 'deps', 'misc'])
                # add sentence labels
                .assign(sent_id = lambda x: (x.id==1).cumsum())
                # replace bad root dependency tags
                .replace({'deprel': {'prd': 'ROOT'}})
               )
    
    if add_head_stuff:
        df = df.merge(df[['id', 'form', 'sent', 'upostag']].rename(index=str, columns={'form': 'head_form', 'upostag': 'head_upos'}).set_index(['sent', 'id']),
               left_on=['sent', 'head'], right_index=True, how='left')
    return df

def parse_sentences(buffer):
    sent = []
    for line in buffer:
        if line.strip() == "":
            if not sent:
                continue
            yield "".join(sent).rstrip()
            sent = []
        else:
            sent.append(line)
    if sent:
        yield "".join(sent).rstrip()


def read_lattice(lattice):
    df = pd.read_csv(StringIO(lattice), sep='\t', header=None, quoting=3, 
                               names = ['ID1', 'ID2', 'form', 'lemma', 'upostag', 'xpostag', 'feats', 'token_id'])
    return df


def read_lattices(filepath_or_buffer):
    if isinstance(filepath_or_buffer, str):
        buffer = open(filepath_or_buffer, 'r', encoding='utf8')
    elif isinstance(filepath_or_buffer, StringIO):
        buffer = filepath_or_buffer
        
    dfs = []
    for i, sent in enumerate(parse_sentences(buffer)):
        dfs.append(read_lattice(sent).assign(sent_id = i+1))
    
    return pd.concat(dfs).reset_index(drop=True)


flatten = lambda l: [item for sublist in l for item in sublist]


def get_feats(s):
    if s!='_' and s is not None and s is not np.nan:
        feats = OrderedDict()
        for f in s.split('|'):
            k,v = f.split('=')
            k='feats_'+k
            if k not in feats:
                feats[k] = v
            else:
                feats[k] = feats[k]+','+v
        return pd.Series(feats)
    else:
        return pd.Series()

    
def read_yap_output(treebank_set=None, tokens_filepath_or_buffer=None, dep_filepath_or_buffer=None, map_filepath_or_buffer=None, expand_feats=False, comment=None):
    if treebank_set is not None:
        tokens_path = TREEBANK_TOKEN_PATHS[treebank_set]
        dep_path = YAP_OUTPUT_PATHS['dep'][treebank_set]
        map_path = YAP_OUTPUT_PATHS['map'][treebank_set]

    if isinstance(tokens_filepath_or_buffer, str):
        tok_buffer = open(tokens_filepath_or_buffer, 'r', encoding='utf8')
    elif isinstance(tokens_filepath_or_buffer, StringIO):
        tok_buffer = tokens_filepath_or_buffer

    tokens = dict(flatten([[(str(j+1)+'_'+str(i+1), tok) for i, tok in enumerate(sent.split('\n'))]
              for j, sent in 
              enumerate(parse_sentences(tok_buffer))]))

    lattices = read_lattices(map_filepath_or_buffer)
    dep = read_conll(dep_filepath_or_buffer, comment=comment)
    df = (pd.concat([dep, lattices.token_id], axis=1)
          .assign(sent_tok = lambda x: x.sent_id.astype(str) + '_' + x.token_id.astype(str))
          .assign(token_str = lambda x: x.sent_tok.map(tokens))
          .drop('sent_tok', axis=1)
          )
    if expand_feats:
        df = pd.concat([df, df.feats.apply(get_feats)], axis=1).drop('feats', axis=1)
        
    return df
