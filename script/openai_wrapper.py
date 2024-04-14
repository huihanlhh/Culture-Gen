import openai
openai.api_key = None
import backoff

class OpenAIWrapper:
    """
        Wrapper for OpenAI API.
    """
    def __init__(self, path):
        self.model_path = path
    
    @backoff.on_exception(backoff.expo, (openai.error.APIError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIConnectionError, openai.error.Timeout))
    def generate(self, prompt=None, temperature=1, max_tokens=512, top_p=1, n=1, get_logprobs=False):
        texts = []
        yes_probs = None
        if self.model_path == "text-davinci-003":
            if get_logprobs:
                yes_probs = []
                response = openai.Completion.create(
                    engine=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                    logprobs=5,
                )
                for choice in response["choices"]:
                    texts.append(choice["text"])
                for choice in response["choices"]:
                    yes_prob = None
                    first_index = 0
                    for i, token in enumerate(choice["logprobs"]["tokens"]):
                        if token != "\n":
                            first_index = i
                            break
                    for token in choice["logprobs"]["top_logprobs"][first_index]:
                        if token.lower().strip() == 'yes':
                            yes_prob = choice["logprobs"]["top_logprobs"][first_index][token]
                            break
                    if yes_prob is None:
                        yes_prob = float('-inf')
                    yes_probs.append(yes_prob)
            else:
                response = openai.Completion.create(
                    engine=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    n=n,
                )
                for choice in response["choices"]:
                    texts.append(choice["text"])
        else:
            response = openai.ChatCompletion.create(
                model=self.model_path,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=n,
            )
            for choice in response["choices"]:
                texts.append(choice['message']['content'])
        return texts, yes_probs