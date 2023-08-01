import random
import pytest  # type: ignore
from dotenv import load_dotenv
from llama2.llama2 import Llama2


@pytest.fixture(scope='session', autouse=True)
def load_env() -> None:
    load_dotenv()


def getWord() -> str:
    return random.choice(
        [
            'foobar',
            'foo',
            'bar',
            'baz',
            'qux',
            'quux',
            'corge',
            'grault',
            'garply',
            'waldo',
            'fred',
            'plugh',
            'xyzzy',
            'thud',
        ]
    )


def test_Llama2() -> None:
    llama2: Llama2 = Llama2()
    word: str = getWord()
    assert llama2.chatCompletion('remember the following word: ' + word)
    assert llama2.chatCompletion('hello world')
    assert llama2.chatCompletion('what was the word I asked you to remember?')
    assert word in llama2.getChat()[-1].assistant
