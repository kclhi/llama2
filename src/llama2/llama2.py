import json, logging, requests, datetime
from dataclasses import asdict

from llama2.llama2_types import Llama2ChatExchange, Llama2Parameters, TokenisedResponse


class Llama2:
    def __init__(
        self,
        apiURL: str = 'http://127.0.0.1:8080',
        chat: list[Llama2ChatExchange] | None = None,
    ) -> None:
        self.__logger: logging.Logger = logging.getLogger()
        self.__apiURL: str = apiURL
        self.__instruction: str = (
            'Complete the following chat script between a human and an artificial intelligence assistant named Llama2. The responses from the human are always prefixed by \'### Human:\'. The assistant (named Llama2) gives helpful, detailed, and polite answers to the human\'s questions. Llama2 was created by Meta. Llama2 runs at King\'s College London. Llama2\'s last training data is from July 2023. The date today is '
            + str(datetime.datetime.now())
        )
        self.__n_keep: int = len(self.__tokenise(self.__instruction))
        self.__chat: list[Llama2ChatExchange] = chat or [
            Llama2ChatExchange(
                human='Hello, Llama2.', assistant='Hello. How may I help you today?'
            ),
            Llama2ChatExchange(
                human='Please tell me the largest city in Europe.',
                assistant='Sure. The largest city in Europe is Moscow, the capital of Russia.',
            ),
        ]

    def getChat(self) -> list[Llama2ChatExchange]:
        return self.__chat

    def __tokenise(self, content: str) -> list[str]:
        try:
            response: requests.Response = requests.post(
                f'{self.__apiURL}/tokenize',
                json={'content': content},
                headers={"Content-Type": "application/json"},
            )
            tokens: TokenisedResponse = response.json(
                object_hook=lambda o: TokenisedResponse(**o)
            )
            return tokens.tokens
        except requests.exceptions.RequestException as e:
            self.__logger.error(f"problem getting tokens: {e}")
            return []

    def __formatPrompt(self, question: str) -> str:
        chat: str = '\n'.join(
            [
                f'### Human: {message.human}\n### Llama2: {message.assistant}'
                for message in self.__chat
            ]
        )
        return f'{self.__instruction}\n{chat}\n### Human: {question}\n### Llama2:'

    def chatCompletion(self, question: str) -> bool:
        params: Llama2Parameters = Llama2Parameters(
            prompt=self.__formatPrompt(question),
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            n_keep=self.__n_keep,
            n_predict=256,
            stop=[
                '\n### Human',
                '###',
                '##',
                '#',
            ],  # stop completion after generating this,
            stream=True,
        )
        self.__logger.debug(params)
        try:
            response: requests.Response = requests.post(
                f'{self.__apiURL}/completion',
                headers={"Content-Type": "application/json"},
                stream=True,
                json=asdict(params),
            )
        except requests.exceptions.RequestException as e:
            self.__logger.error(f"problem getting completion: {e}")
            return False

        answer: str = ''
        chunk: bytes
        for chunk in response.iter_content(chunk_size=8192):
            t: str = chunk.decode('utf-8')
            if t.startswith('data: '):
                message: dict[str, str] = json.loads(t[6:])
                answer += message['content']
                if message.get('stop', False):
                    if message.get('truncated', False):
                        self.__chat.pop()
                    break

        self.__chat.append(Llama2ChatExchange(human=question, assistant=answer.strip()))
        self.__logger.debug(answer.strip())
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    llama2: Llama2 = Llama2()
    llama2.chatCompletion('what is your name?')
    print(llama2.getChat()[-1].assistant)
