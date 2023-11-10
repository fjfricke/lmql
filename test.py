import sys
sys.path.append('/Users/felix/Programming/lmql/src')
import lmql
import asyncio

#add replicate api key to env
import os
os.environ['REPLICATE_API_TOKEN'] = 'r8_aOlrg82Wfg30Rx4L4mv9wI2npPfBQGO0Pvci4'

# def test_decorator(variable_value, prompt_value, context):
#     return variable_value, prompt_value

async def main():

    test = lmql.model(
        "openai/gpt-3.5-turbo-instruct"
        # "meta-llama/Llama-2-13b-chat-hf",
        # endpoint="replicate:deployment/ml-delphai/llama2-13b-chat-lmtp",
        # endpoint="replicate:charles-dyfis-net/llama-2-7b-chat-hf--lmtp-8bit",
        # tokenizer="AyyYOO/Luna-AI-Llama2-Uncensored-FP16-sharded",
    )
    pass

    answer = await lmql.run(
        """
        import math
        def get_probs(variable_value, prompt_value, context):
            breakpoint()
            logprob_scores = list(context.variable_scores.items())[-1][1]
            scores = dict()
            for key, value in logprob_scores.items():
                if value > -5:
                    scores[key] = math.exp(value)
            return scores
        argmax(verbose=True)
        \"How much you like monkeys between 0 and 2?[@get_probs MONKEY]\" where MONKEY in set ([\"0\", \"1\", \"2\"])
        \"How much you like birds between 0 and 2?[@get_probs BIRD]\" where BIRD in set ([\"0\", \"1\", \"2\"])
        return (MONKEY, BIRD)
        """,
        max_len=4000,
        model=test,
        # decoder_graph=True,
    )

    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
