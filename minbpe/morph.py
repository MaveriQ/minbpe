"""
Implements the BPE Tokenizer with byte-fallback and coverage options (LlamaTokenizer?) as a light wrapper around the RegexTokenizer.
This tokenizer needs to be trained on a text corpus before it can be used for encoding/decoding.
"""

import pickle
import string
from .regex import RegexTokenizer
from collections import Counter, OrderedDict
import regex as re
from tqdm import tqdm
import os
from pathlib import Path

cur_dir = Path(os.path.dirname(__file__))

LLAMA_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # gpt4 with individual digits split

LLAMA_SPECIAL_TOKENS = { # added pad token
    '<|unk|>': 0,
    '<s>': 1,
    '</s>': 2,
    '<|pad|>': 3,
}

class LlamaMorphPiece(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches (?) Llama's tokenizer."""

    def __init__(self, character_coverage = 0.9995, 
                 pattern=LLAMA_SPLIT_PATTERN, special_tokens=LLAMA_SPECIAL_TOKENS,
                 morpheme_file=cur_dir/'vocab_v2/new_lookup_table.pkl'):
        super().__init__(pattern=pattern)

        # register the special tokens
        self.register_special_tokens(special_tokens)
        self.unk_token_id = self.special_tokens['<|unk|>']
        self.num_special_tokens = len(self.special_tokens)
        
        self.character_coverage = character_coverage
        self.vocab = "deprecated. Use vocab_itos or vocab_stoi" # overwrites the vocab from the base class to use custom __build_vocab function
        # self.merges = {} # available from the base class
        self.rare_tokens = []
        self.counts = {}
        self.verbose = False
        self.morphtable,self.old_vocab = pickle.load(open(morpheme_file,'rb'))
        
    def build_morphvocab(self,text):
        
        #TODO Include only those entries in morphtable which are present in the training corpus
        # Fix problematic morphtable entries        
        morphemes_to_fix = ['dewy','snuffy','toasty','weasely','tonguey'] # problematic detokenizations with combinations like "dewy drops" and "toasty pulsating"
        # 'soppy' moved to morphtable as ['sop','#y']

        for morpheme in morphemes_to_fix:
            self.morphtable[morpheme] = [morpheme[:-1],"#" + morpheme[-1]]
                
        # Build the vocab
        flatten = lambda l: [item for sublist in l for item in sublist]
        morphvocab = sorted(set(flatten(self.morphtable.values())))
        
        for i,x in enumerate(self.old_vocab): # Confirming that old and sorted vocab are same lists
            assert x==morphvocab[i]
            
        return morphvocab
            
    def __build_vocab(self,text): # to differentiate from the base class's _build_vocab. Used only in training (without merges)
        
        # Build rare_tokens from coverage
        vocab_counts = Counter(text)
        freq_sort = sorted(vocab_counts.items(), key=lambda item: (-item[1], item[0])) # sort by freq (item[1]) and break ties alphabetically (item[0]).
        freq_sort = OrderedDict(freq_sort)
            
        total_chars = len(text)
        coverage_count = int((1-self.character_coverage) * total_chars)
        rare_tokens = [char for char,freq in freq_sort.items() if freq<coverage_count]
        unicode_vocab = [char for char,freq in freq_sort.items() if freq>=coverage_count]
    
        vocab_itos = {v:k for k,v in LLAMA_SPECIAL_TOKENS.items()} # start with special tokens
        vocab_itos.update({idx+self.num_special_tokens: bytes([idx]) for idx in range(256)}) # add utf-8 fallback bytes
        vocab_itos.update({idx+self.num_special_tokens + 256: ch for idx,ch in enumerate(unicode_vocab)}) # add unicode vocabulary. Merges are added during traning. 
        return vocab_itos, rare_tokens
    
    def get_token_ids(self,text,vocab_itos=None, stage='eval'):
        vocab_itos = vocab_itos if vocab_itos is not None else self.vocab_itos # For training we pass the vocab_wo_merges. For encoding we use the trained self.vocab_itos.
        vocab_stoi = {v:k for k,v in vocab_itos.items()} if vocab_itos is not None else self.vocab_stoi

        token_ids = []
        for ch in text:
            if ch in self.rare_tokens:
                token_ids.append(self.unk_token_id)
            elif ch in vocab_stoi:
                token_ids.append(vocab_stoi[ch])
            else: # utf-8 fallback
                if self.verbose: print(f'utf-8 encoded : {ch}')
                x_utf8 = ch.encode("utf-8")
                x_utf8_token_ids = [i+self.num_special_tokens for i in x_utf8] # offset special_tokens
                token_ids.extend(x_utf8_token_ids)
                
        if stage=='train': # no token_id should be from fallback utf-8 during training
            assert not any(x in range(self.num_special_tokens,self.num_special_tokens+256) for x in token_ids)
        return token_ids
    
    def get_counts_llama(self,ids,counts=None): # renamed stats to counts
        """
        Given a list of integers, updates a dictionary of counts of consecutive pairs.
        unk_tokens are arbitrarily given 0 count. (Ignoring them leads to issues with the merge step.)
        Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
        Optionally allows to update an existing dictionary of counts
        """
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            if self.unk_token_id in pair:
                counts[pair] = 0
            else:
                counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge_llama(self,ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            
            # if unk_token_id, skip over it
            if ids[i] == self.unk_token_id:
                newids.append(ids[i])
                i += 1
            # if not at the very last position AND the pair matches, replace it
            elif ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self,text,required_vocab_size,verbose=None,coverage=None):
        self.verbose = verbose if verbose is not None else self.verbose
        self.character_coverage = coverage if coverage is not None else self.character_coverage
        
        # build the vocab. Determine number of merges
        vocab_wo_merges, self.rare_tokens = self.__build_vocab(text)
        vocab_len = len(vocab_wo_merges)
        assert required_vocab_size >= vocab_len, f"Required vocab size {required_vocab_size} is less than the vocab size {vocab_len}"
        num_merges = required_vocab_size - vocab_len
        if self.verbose: print(f"New Merges : {num_merges}")
        
        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)
        
        # Get Unicode codepoints for each chunk. Replace the characters in unk_tokens with unk_token_id.
        ids = [self.get_token_ids(chunk,vocab_itos=vocab_wo_merges,stage='train') for chunk in text_chunks] # unicode encoded
        
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        
        for i in tqdm(range(num_merges),total=num_merges,disable=self.verbose):
            # count the number of times every consecutive pair appears
            counts = {}
            for chunk_ids in ids:
                # passing in counts will update it in place, adding up counts
                self.get_counts_llama(chunk_ids,counts)
            # find the pair with the highest count
            pair = max(counts, key=counts.get)
            # mint a new token: assign it the next available id
            idx = vocab_len + i
            # replace all occurrences of pair in ids with idx
            ids = [self.merge_llama(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab_wo_merges[idx] = vocab_wo_merges[pair[0]] + vocab_wo_merges[pair[1]] 
            
            if self.verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab_wo_merges[idx]}) had {counts[pair]} occurrences")

        # adding morphpiece vocabulary
        morphvocab = self.build_morphvocab(text)
        for token in morphvocab:
            if token not in vocab_wo_merges.values():
                vocab_wo_merges[len(vocab_wo_merges)] = token
                    
        # save class variables
        self.merges = merges # used in encode()
        self.vocab_itos = vocab_wo_merges   # used in decode()
        
    def _encode_chunk(self, tokens):
        # given a string, return list of integers (the tokens)

        while len(tokens) >= 2:
            counts = self.get_counts_llama(tokens)
            pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                if self.verbose: print(f"breaking out on pair : {pair}")
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge_llama(tokens, pair, idx)
        return tokens
    
    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            
            chunk_without_space = chunk.strip()
            
            morph_splits = self.morphtable.get(chunk_without_space,chunk)
            
            if isinstance(morph_splits,list): # We got morphemes.
                morph_token_ids=[]
                for morpheme in morph_splits:
                    morph_token_ids.append(self.vocab_stoi[morpheme])
                ids.extend(morph_token_ids)
            else:
                assert chunk == morph_splits
                # Get Unicode codepoints for each chunk. Replace the characters in unk_tokens with unk_token_id.
                tokens = self.get_token_ids(chunk)
                try:
                    chunk_ids = self._encode_chunk(tokens)
                except Exception as e:
                    print(f"Error in chunk : {chunk}")
                    raise e
                ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
    
    def decode(self,ids):
        tokens = self._convert_ids_to_tokens(ids)
        converted_text = self.convert_tokens_to_string(tokens)
        return converted_text
        
    def _convert_ids_to_tokens(self,ids):
        
        def flush_chunk(tokens,chunk,typ):
            if typ=='utf8':
                part_bytes = []
                for idx in chunk:
                    if idx in self.inverse_special_tokens:
                        tokens.append(self.inverse_special_tokens[idx])
                    else:
                        part_bytes.append(self.vocab_itos[idx])
                tokens.append(b"".join(part_bytes).decode())
            elif typ=='uni': # also for morphpiece tokens
                tokens += [self.vocab_itos[idx] for idx in chunk]
            chunk = []
            return tokens,chunk
        
        tokens = []
        utf8_chunk=[]
        uni_chunk=[]
        
        utf8_boundary = self.num_special_tokens + 256
        
        # ids are of two types, from utf-8 bytes/special tokans and unicode/morphpiece tokens. 
        # I iterate over ids and flush out the chunks as needed, updating the output text string. 
        
        for id in ids:
            if id<utf8_boundary: # utf-8 fallback bytes and special tokens except for unk_token
                if len(uni_chunk)>0:
                    tokens,uni_chunk = flush_chunk(tokens,uni_chunk,'uni')
                utf8_chunk.append(id)
            else: # unicode characters merges and morphpiece tokens
                if len(utf8_chunk)>0:
                    tokens,utf8_chunk = flush_chunk(tokens,utf8_chunk,'utf8')
                uni_chunk.append(id)
        #flush out the last chunk
        if len(utf8_chunk)>0:
            tokens,_ = flush_chunk(tokens,utf8_chunk,'utf8')
        elif len(uni_chunk)>0:
            tokens,_ = flush_chunk(tokens,uni_chunk,'uni')
        return tokens

    # save/load doesn't have the issues of the GPT4Tokenizer as there is no shuffling involved
    def save(self, model_file):
        payload = {'vocab_itos': self.vocab_itos, 
                   'merges': self.merges, 
                   'special_tokens': self.special_tokens,
                   'rare_tokens': self.rare_tokens,
                   'character_coverage': self.character_coverage,
                   'pattern': self.pattern,
                   }
        
        pickle.dump(payload, open(model_file, "wb"))

    def load(self, model_file):
        payload = pickle.load(open(model_file, "rb"))
        self.vocab_itos = payload['vocab_itos']
        self.merges = payload['merges']
        self.special_tokens = payload['special_tokens']
        self.rare_tokens = payload['rare_tokens']
        self.character_coverage = payload['character_coverage']
        self.pattern = payload['pattern']
        self.num_special_tokens = len(self.special_tokens)
        self.vocab_stoi = {v:k for k,v in self.vocab_itos.items()}

    @classmethod
    def from_pretrained(cls, model_file):
        tokenizer = cls()
        tokenizer.load(model_file)
        return tokenizer
    
    def get_token_type(self,tokens):
        token_type = []
        for i,token in enumerate(tokens):
            if token=='#': # compound word
                if not ("Ġ" in tokens[i-1] or "#" in tokens[i-1]): # check if previous token is a bpe or a suffix
                    # Rewriting the affix of the previous token if it's not already bpe or a suffix
                    token_type[i-1] = 'stem'
                token_type.append('#')
            elif token.startswith('#'):
                if not (tokens[i-1].startswith('Ġ') or tokens[i-1].startswith('#')): 
                    # Rewriting the affix of the previous token if it's not already bpe or a suffix
                    token_type[i-1] = 'stem'
                token_type.append('suffix')
            elif token.startswith('Ġ') or token.startswith(' '):
                token_type.append('bpe')
            elif token.endswith('#'):
                token_type.append('prefix')
            else: # When there is no special character (# or Ġ) in the token

                pat_punct = re.compile("[" + re.escape(string.punctuation) + "]+")
                pat_num = r'\d+' # Pattern for finding numbers
                
                if len(re.findall(pat_punct, token))!=0: # If punctuation is found
                    token_type.append('bpe') # mark the token as bpe
                elif len(re.findall(pat_num, token))!=  0: # If digits are found
                    token_type.append('bpe') # mark the token as bpe
                elif i==0: # For the first token
                    if len(tokens)==1: # If there is only one token
                        token_type.append('bpe') # mark the token as bpe
                    else:
                        if tokens[i+1].startswith('#'): # Stem if next token is a suffix
                            token_type.append('stem')
                        else: # otherwise it's BPE
                            token_type.append('bpe')
                elif token_type[i-1]=='bpe':
                    token_type.append('bpe')
                else:
                    token_type.append('stem')

        self.token_type = token_type

    def process_prefix(self,tokens,idx=0):
        if idx+1==len(tokens):
            return [tokens[idx]], idx
        if self.token_type[idx+1]=='bpe':
            return [tokens[idx]], idx
        elif self.token_type[idx+1] == 'prefix':
            word,final_idx = self.process_prefix(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word,final_idx
        elif self.token_type[idx+1] == 'stem':
            word, final_idx = self.process_stem(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx
        else:
            raise Exception(f"There is no handler for {self.token_type[idx+1]} in process_prefix")
            
    def process_suffix(self,tokens,idx=0):
        if idx+1==len(tokens):
            return [tokens[idx]], idx
        if self.token_type[idx+1] in ['bpe','stem', 'prefix']:
            return [tokens[idx]], idx
        elif self.token_type[idx+1] in ['suffix']:
            word, final_idx = self.process_suffix(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx
        elif self.token_type[idx+1] in ['#']:
            word, final_idx = self.process_hash(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx
        else:
            raise Exception(f"There is no handler for {self.token_type[idx+1]} in process_suffix")
            
    def process_stem(self,tokens,idx=0):
        if idx+1==len(tokens):
            return [tokens[idx]], idx
        if self.token_type[idx+1] in ['bpe','prefix']:
            return [tokens[idx]], idx
        elif self.token_type[idx+1] in ['stem']:
            if len(tokens[idx])==1: # #FIXME Using the hack that compound words with multiple stems start with single letter token
                word, final_idx = self.process_stem(tokens,idx+1)
                word.insert(0,tokens[idx])
                return word, final_idx
            else: # next word
                return [tokens[idx]], idx
        elif self.token_type[idx+1] in ['#']:
            word, final_idx = self.process_hash(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx        
        elif self.token_type[idx+1] in ['suffix']:
            word, final_idx = self.process_suffix(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx
        else:
            raise Exception(f"There is no handler for {self.token_type[idx+1]} in process_stem")
        
    def process_hash(self,tokens,idx=0): # compound words

        if self.token_type[idx+1] in ['stem']:
            word, final_idx = self.process_stem(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx
        elif self.token_type[idx+1] in ['prefix']: # (god,#,for#,sake,#n)
            word, final_idx = self.process_prefix(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx
        else:
            raise Exception(f"There is no handler for {self.token_type[idx+1]} in process_hash")
            
    def process_bpe(self,tokens,idx=0):
        if idx+1==len(tokens):
            return [tokens[idx]], idx
        if self.token_type[idx+1] in ['stem','prefix']:
            return [tokens[idx]], idx
        elif self.token_type[idx+1] in ['bpe']:
            word, final_idx = self.process_bpe(tokens,idx+1)
            word.insert(0,tokens[idx])
            return word, final_idx
        else:
            raise Exception(f"There is no handler for {self.token_type[idx+1]} in process_bpe")

    def _convert_bpe_tokens_to_string(self, tokens):
        """Converts a sequence of BPE tokens (string) in a single string."""
        text = "".join(tokens)
        # text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def convert_tokens_to_string(self,tokens):
        
        self.get_token_type(tokens)

        splits = []
        sentence = []
        i=0
        while i<len(tokens):
            if self.token_type[i]=='prefix':
                word,i = self.process_prefix(tokens,i)
                source = 'morphpiece'
            elif self.token_type[i]=='suffix':
                word,i = self.process_suffix(tokens,i)
                source = 'morphpiece'
            elif self.token_type[i]=='stem':
                word,i = self.process_stem(tokens,i)
                source = 'morphpiece'
            elif self.token_type[i]=='bpe':
                word,i = self.process_bpe(tokens,i)
                source = 'bpe'
            elif self.token_type[i]=='#': # lone-# not part of a compound word
                word,i = tokens[i],i
                source = 'bpe'
            else:
                raise(ValueError)
            splits.append((source,tuple(word)))
            i+=1

        for split in splits:
            if split[0]=='bpe':
                shard = self._convert_bpe_tokens_to_string(split[1]).strip()
            else:
                shard = self.reverse_morphtable.get(split[1],split[1][0])
            sentence.append(shard)

        return " ".join(sentence)