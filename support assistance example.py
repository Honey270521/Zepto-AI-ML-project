import json
from app import ask
for q in [
    'What is the delivery fee?',
    'What are support hours?',
    'Tell me a joke.'
]:

    print(
        'QUERY:',
        q
    )

    print(
        json.dumps(
            ask(q).model_dump(),
            indent=2
        )
    )