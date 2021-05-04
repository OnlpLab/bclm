import pandas as pd
import unicodedata
from .hebtokenizer import tokenize as hebtokenize
from .hebtokenizer import alternative_scanner

TOK_FIELDS = ['sent_id', 'token_id', 'token_str']


def get_token_biose(df, biose_field, token_fields = TOK_FIELDS):
    
    def _single_token_conversion(tok):
        all_bio = tok.biose_only.tolist()
        all_typ = set(tok.ner_type.dropna().tolist())
        if len(all_typ)>1:
            return 'O'
        if 'S' in all_bio:
            new_bio = 'S'
        elif 'B' in all_bio and 'E' in all_bio:
            new_bio = 'S'
        elif 'B' in all_bio:
            new_bio = 'B'
        elif 'E' in all_bio:
            new_bio = 'E'
        elif 'I' in all_bio:
            new_bio = 'I'
        else:
            return 'O'
        return new_bio+'-'+all_typ.pop()
    
    
    split_biose = df[biose_field].str.split('-', expand=True)
    if len(split_biose.columns)==2:
        df[['biose_only', 'ner_type']] = split_biose
    else:
        df['biose_only'] = split_biose
        df['ner_type'] = None
        
    df = (df
          .groupby(token_fields)
          .apply(_single_token_conversion)
          .reset_index().rename(columns={0: biose_field})
          .set_index(token_fields)
         )
    return df


def get_token_df(df, fields=None, biose=None, token_fields = TOK_FIELDS, sep='^', fill_value='', add_set=True):
    tok_dfs = []
    
    if biose is not None:
        for col in biose:
            tok_dfs.append(get_token_biose(df, col))
        
    if fields is not None:
        concat_fields = lambda x: pd.Series({f: sep.join(x[f].fillna(fill_value).tolist()) for f in fields})
        tok_fields = (df
                .groupby(token_fields)
                .apply(concat_fields))
        tok_dfs.append(tok_fields)
        
    tok_df = pd.concat(tok_dfs, axis=1)

    if add_set and 'set' in df.columns:
            tok_df = tok_df.assign(set = lambda x: (x.index
                                                     .get_level_values('sent_id')
                                                     .map(df[['sent_id', 'set']]
                                                     .drop_duplicates()
                                                     .set_index('sent_id')['set'])))
            
    tok_df = tok_df.sort_index().reset_index()
    
    return tok_df


def get_sentences_list(df, fields, sent_id='sent_id', drop_yy=False):
    if drop_yy:
        out_df = df[~df.upostag.str.startswith('yy')]
    else:
        out_df = df
    return out_df.groupby(sent_id)[fields].apply(lambda x: (x.values.tolist()))


def get_feature_lists(df, fields, sent_id='sent_id'):
    feats = []
    for field in fields:
        feats.append(df.groupby(sent_id)[field].apply(lambda x: (x.values.tolist())))
    return feats


### Tokenization

def clean_hebchars(text):
    norm = unicodedata.normalize('NFKD', text)
    text =''.join([c for c in norm if not unicodedata.combining(c)]) 
    #line = line.replace('־', '-')
    # maqaf
    text = text.replace(u'\u05be', '-')

    #line = line.replace('׳', '\'')
    #geresh
    text = text.replace(u'\u05f3', '\'')

    #line = line.replace('״', '"')
    #gershayim
    text = text.replace(u'\u05f4', '"')
    #line = line.replace('”', '"')
    #line = line.replace('„', '"')
    #en dash
    text = text.replace(u'\u2013', '-')
    #em dash
    text = text.replace(u'\u2014', '-')
    return text

def tokenize(sent, alt_scan=True, clean_junk=True, clean_heb_chars=True):
    if clean_heb_chars:
        sent = clean_hebchars(sent.strip())
    if alt_scan:
        tok = hebtokenize(sent, alternative_scanner)
    else:
        tok = hebtokenize(sent)

    last_type, last_form = tok[-1]
    if len(last_form)>1 and last_type!='PUNCT' and last_form[-1] in ('?', '!', '.'):
        tok[-1] = (last_type, last_form[:-1])
        tok.append(('PUNCT', last_form[-1]))

    final = []
    for c, t in tok: 
        if clean_junk and c=='JUNK':
            continue
        final.append(t)
    
    return final