"""
Tests mixing different tokenizers/models in the same LMQL process.
"""

import lmql
from lmql.tests.expr_test_utils import run_all_tests

RANDOM_GPT_OUTPUT = "saf steppingattery clutch"
RANDOM_LLAMA_OUTPUT = "jlhidePhotoSR"

@lmql.query(model=lmql.model("random", tokenizer="gpt2", seed=1))
async def test_random_gpt():
    '''lmql
    "Hello[WORLD]" where len(TOKENS(WORLD)) == 4
    assert WORLD == RANDOM_GPT_OUTPUT, "Expected '{}', got '{}'".format(
        RANDOM_GPT_OUTPUT,
        WORLD
    )
    return WORLD
    '''

@lmql.query(model=lmql.model("random", tokenizer="AyyYOO/Luna-AI-Llama2-Uncensored-FP16-sharded", seed=1))
async def test_random_llama():
    '''lmql
    "Hello[WORLD]" where len(TOKENS(WORLD)) == 4
    assert WORLD == RANDOM_LLAMA_OUTPUT, "Expected '{}', got '{}'".format(
        RANDOM_LLAMA_OUTPUT,
        WORLD
    )
    return WORLD
    '''

@lmql.query(model=lmql.model("random", tokenizer="gpt2", seed=1))
async def test_llama_from_gpt():
    '''lmql
    "Hello[WORLD]" where len(TOKENS(WORLD)) == 4
    assert [WORLD, test_random_llama()] == [
        RANDOM_GPT_OUTPUT,
        RANDOM_LLAMA_OUTPUT
    ], "Expected {}, got {}".format(
        [RANDOM_GPT_OUTPUT, RANDOM_LLAMA_OUTPUT],
        [WORLD, test_random_llama()]
    )
    '''

if __name__ == "__main__":
    run_all_tests(globals())