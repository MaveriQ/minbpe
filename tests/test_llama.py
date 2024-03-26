from minbpe import LlamaTokenizer
import pytest
import os

# -----------------------------------------------------------------------------
# common test data

# a few strings to test the tokenizers on
test_strings = [
    "", # empty string
    "?", # single character
    "hello world!!!? (안녕하세요!) lol123 😉", # fun small string
    "FILE:taylorswift.txt", # FILE: is handled as a special string in unpack()
]
def unpack(text):
    # we do this because `pytest -v .` prints the arguments to console, and we don't
    # want to print the entire contents of the file, it creates a mess. So here we go.
    if text.startswith("FILE:"):
        dirname = os.path.dirname(os.path.abspath(__file__))
        taylorswift_file = os.path.join(dirname, text[5:])
        contents = open(taylorswift_file, "r", encoding="utf-8").read()
        return contents
    else:
        return text

specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()
special_tokens = {
    '<|unk|>': 0,
    '<s>': 1,
    '</s>': 2,
    '<|pad|>': 3,
}

llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

@pytest.mark.parametrize("text", test_strings)
def test_encode_decode_identity(text):
    text = unpack(text)
    tokenizer = LlamaTokenizer(character_coverage=1.0) # for coverage < 1.0, we get unk_tokens, and the decoded text would not be the same as the original
    tokenizer.train(llama_text, 400)
    ids = tokenizer.encode_ordinary(text)
    decoded = tokenizer.decode(ids)
    assert text == decoded
    
# reference test to add more tests in the future
def test_wikipedia_example():
    """
    Quick unit test, following along the Wikipedia example:
    https://en.wikipedia.org/wiki/Byte_pair_encoding

    According to Wikipedia, running bpe on the input string:
    "aaabdaaabac"

    for 3 merges will result in string:
    "XdXac"

    where:
    X=ZY
    Y=ab (or Za in our case, since both have same counts)
    Z=aa

    Keep in mind that for us special + utf-8 fallback offset = 4+256=260, so a=260, b=261, c=262, d=263 (unicode values)
    so Z will be 264, Y will be 265, X will be 266.

    So we expect the output list of ids to be [266, 263, 266, 260, 262] for [XdXac]
    """
    tokenizer = LlamaTokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, 4 + 256 + 4 + 3) # 4 special tokens, 256 fallback, 4 unicode vocab (a-d), 3 merges 
    ids = tokenizer.encode(text)
    assert ids == [266, 263, 266, 260, 262]
    assert tokenizer.decode(tokenizer.encode(text)) == text

@pytest.mark.parametrize("special_tokens", [{}, special_tokens])
def test_save_load(special_tokens):
    # take a bit more complex piece of text and train the tokenizer, chosen at random
    text = llama_text
    # create a Tokenizer for 400 vocab size, and train it on the text
    tokenizer = LlamaTokenizer(character_coverage=1.0) # to prevent unk_tokens
    tokenizer.train(text, 400,verbose=False) # 256 fallback, 4 special tokens, 64 merges
    tokenizer.register_special_tokens(special_tokens)
    # verify that decode(encode(x)) == x
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    # verify that save/load work as expected
    ids = tokenizer.encode(text, "all")
    # save the tokenizer (TODO use a proper temporary directory)
    tokenizer.save("test_tokenizer_tmp.model")
    # re-load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained("test_tokenizer_tmp.model")
    # tokenizer.load("test_tokenizer_tmp.model")
    # verify that decode(encode(x)) == x
    assert tokenizer.decode(ids) == text
    assert tokenizer.decode(tokenizer.encode(text, "all")) == text
    assert tokenizer.encode(text, "all") == ids
    # delete the temporary files
    for file in ["test_tokenizer_tmp.model"]:
        os.remove(file)

if __name__ == "__main__":
    pytest.main()