"""
Implements the BPE Tokenizer with byte-fallback and coverage options (LlamaTokenizer?) as a light wrapper around the RegexTokenizer.
This tokenizer needs to be trained on a text corpus before it can be used for encoding/decoding.
"""

import pickle
from .regex import RegexTokenizer
from collections import Counter, OrderedDict
import regex as re
from tqdm import tqdm

LLAMA_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""" # gpt4 with individual digits split

LLAMA_SPECIAL_TOKENS = { # added pad token
    '<|unk|>': 0,
    '<s>': 1,
    '</s>': 2,
    '<|pad|>': 3,
}

class LlamaTokenizer(RegexTokenizer):
    """Lightweight wrapper on RegexTokenizer that matches (?) Llama's tokenizer."""

    def __init__(self, character_coverage = 0.9995):
        super().__init__(pattern=LLAMA_SPLIT_PATTERN)

        # register the special tokens
        self.register_special_tokens(LLAMA_SPECIAL_TOKENS)
        self.unk_token_id = self.special_tokens['<|unk|>']
        self.num_special_tokens = len(self.special_tokens)
        
        self.character_coverage = character_coverage
        self.vocab = {} # overwrites the vocab from the base class to have custom build_vocab function
        # self.merges = {} # available from the base class
        self.rare_tokens = []
        self.counts = {}
        self.verbose = False

    def __build_vocab(self,text): # to differentiate from the base class's _build_vocab
        
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
        vocab_itos = vocab_itos if vocab_itos is not None else self.vocab # For training we need the vocab_wo_merges. For encoding we use the trained self.vocab.
        vocab_stoi = {v:k for k,v in vocab_itos.items()}

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
    
        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab_wo_merges   # used in decode()
    
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
            # Get Unicode codepoints for each chunk. Replace the characters in unk_tokens with unk_token_id.
            tokens = self.get_token_ids(chunk)
            try:
                chunk_ids = self._encode_chunk(tokens)
            except Exception as e:
                print(f"Error in chunk : {chunk}")
                raise e
            ids.extend(chunk_ids)
        return ids

    def decode(self,ids, include_unk=False):
        
        def flush_chunk(text,chunk,typ):
            if typ=='utf8':
                tokens = b"".join(self.vocab[idx] for idx in chunk)
                text += tokens.decode("utf-8", errors="replace")
            elif typ=='uni':
                text += "".join(self.vocab[idx] for idx in chunk)
            else: # unk_chunk
                text += '<|unk|>'
            chunk = []
            return text,chunk
        
        text=""
        utf8_chunk=[]
        uni_chunk=[]
        
        utf8_boundary = self.num_special_tokens + 256
        
        # ids are of three types, from utf-8 bytes, unicode characters, and unk_tokens. 
        # I iterate over ids and flush out the chunks as needed, updating the output text string. 
        
        for id in ids:
            if id==0:
                if len(utf8_chunk)>0: # flush out the utf8_chunk and uni_chunk.
                    text,utf8_chunk = flush_chunk(text,utf8_chunk,'utf8')
                elif len(uni_chunk)>0:
                    text,uni_chunk = flush_chunk(text,uni_chunk,'uni')
                    
                if include_unk: text += '<|unk|>'
            elif id<utf8_boundary: # utf-8 fallback bytes
                if len(uni_chunk)>0:
                    text,uni_chunk = flush_chunk(text,uni_chunk,'uni')
                utf8_chunk.append(id)
            else: # unicode characters and merges
                if len(utf8_chunk)>0:
                    text,utf8_chunk = flush_chunk(text,utf8_chunk,'utf8')
                uni_chunk.append(id)
        #flush out the last chunk
        if len(utf8_chunk)>0:
            text,_ = flush_chunk(text,utf8_chunk,'utf8')
        elif len(uni_chunk)>0:
            text,_ = flush_chunk(text,uni_chunk,'uni')
        return text

    # save/load doesn't have the issues of the GPT4Tokenizer as there is no shuffling involved
    def save(self, model_file):
        payload = {'vocab': self.vocab, 
                   'merges': self.merges, 
                   'special_tokens': self.special_tokens,
                   'rare_tokens': self.rare_tokens,}
        
        pickle.dump(payload, open(model_file, "wb"))

    def load(self, model_file):
        payload = pickle.load(open(model_file, "rb"))
        self.vocab = payload['vocab']
        self.merges = payload['merges']
        self.special_tokens = payload['special_tokens']
        self.rare_tokens = payload['rare_tokens']
        self.num_special_tokens = len(self.special_tokens)
