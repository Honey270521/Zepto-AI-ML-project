from __future__ import annotations
import os
import json
import re
import hashlib
from pathlib import Path
from typing import TypedDict
from pydantic import BaseModel, Field
ROOT=Path(__file__).resolve().parent
DOCS=ROOT/'docs'
STORE=ROOT/'data'
STORE.mkdir(
    exist_ok=True
)

MOCK_LLM=os.getenv(
    'MOCK_LLM',
    '1'
)!='0'
try:

    from sentence_transformers import SentenceTransformer

except Exception:

    SentenceTransformer=None

try:

    import chromadb

except Exception:

    chromadb=None

try:

    from langgraph.graph import (
        StateGraph,
        START,
        END
    )

except Exception:

    START='__start__'

    END='__end__'

    class StateGraph:

        def __init__(
            self,
            state_type
        ):
            self.nodes={}
            self.edges={}
            self.conditional={}

        def add_node(
            self,
            name,
            fn
        ):
            self.nodes[name]=fn

        def add_edge(
            self,
            a,
            b
        ):
            self.edges[a]=b

        def add_conditional_edges(
            self,
            a,
            fn,
            mapping
        ):
            self.conditional[a]=(
                fn,
                mapping
            )

        def compile(self):

            outer=self

            class App:

                def invoke(
                    self,
                    state
                ):

                    cur=outer.edges.get(
                        START
                    )

                    s=dict(state)

                    while cur and cur!=END:

                        s.update(
                            outer.nodes[cur](s)
                        )

                        if cur in outer.conditional:

                            f,m=outer.conditional[cur]

                            cur=m[f(s)]

                        else:

                            cur=outer.edges.get(cur)
                        return s
                        return App()
                    


class Answer(BaseModel):

    answer:str

    sources:list[str]=Field(
        default_factory=list
    )

    confidence:float=Field(
        ge=0,
        le=1
    )


class AskRequest(BaseModel):

    query:str


class State(
    TypedDict,
    total=False
):

    query:str

    intent:str

    retrieved:list[dict]

    answer:dict


KEYWORDS=[
    'delivery',
    'return',
    'refund',
    'membership',
    'tracking',
    'cancel',
    'gift card',
    'support hours'
]


def load_docs():

    return [
        {
            'id':p.stem,
            'text':p.read_text(
                encoding='utf-8'
            )
        }

        for p in sorted(
            DOCS.glob(
                'doc_*.txt'
            )
        )
    ]


DOCS_DATA=load_docs()


def embed_text(text):

    if SentenceTransformer:

        model=SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

        return model.encode(
            [text],
            normalize_embeddings=True
        )[0]

    v=[0.0]*128

    for tok in re.findall(
        r'\w+',
        text.lower()
    ):

        i=int(
            hashlib.sha256(
                tok.encode()
            ).hexdigest(),
            16
        )%128

        v[i]+=1

    norm=sum(
        x*x
        for x in v
    )**.5 or 1

    return [
        x/norm
        for x in v
    ]


def build_store():

    if chromadb and SentenceTransformer:

        client=chromadb.PersistentClient(
            path=str(
                STORE/'chroma'
            )
        )

        col=client.get_or_create_collection(
            'zepto_policy'
        )

        model=SentenceTransformer(
            'all-MiniLM-L6-v2'
        )

        ids=[
            d['id']
            for d in DOCS_DATA
        ]

        texts=[
            d['text']
            for d in DOCS_DATA
        ]

        embs=model.encode(
            texts,
            normalize_embeddings=True
        ).tolist()

        col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embs
        )

        return (
            'chroma',
            col
        )

    embs={
        d['id']:
            embed_text(
                d['text']
            )
        for d in DOCS_DATA
    }

    (
        STORE/'fallback_embeddings.json'
    ).write_text(
        json.dumps(embs)
    )

    return (
        'fallback',
        embs
    )


STORE_KIND,VECTOR_STORE=build_store()


def cosine(a,b):

    return sum(
        x*y
        for x,y in zip(a,b)
    )


def retrieve(
    query,
    k=3
):

    if STORE_KIND=='chroma':

        q=embed_text(
            query
        )

        r=VECTOR_STORE.query(
            query_embeddings=[
                q
            ],
            n_results=k
        )

        return [
            {
                'id':i,
                'text':t
            }

            for i,t in zip(
                r['ids'][0],
                r['documents'][0]
            )
        ]

    q=embed_text(
        query
    )

    qwords=set(
        re.findall(
            r'\w+',
            query.lower()
        )
    )

    def score(d):

        words=set(
            re.findall(
                r'\w+',
                d['text'].lower()
            )
        )

        return (
            len(
                qwords & words
            )*10
            +
            cosine(
                q,
                VECTOR_STORE[d['id']]
            )
        )

    scored=sorted(
        (
            (
                score(d),
                d
            )
            for d in DOCS_DATA
        ),
        reverse=True,
        key=lambda x:x[0]
    )

    return [
        d
        for _,d in scored[:k]
    ]


PROMPT_TEMPLATE='''ROLE: You are a Zepto policy support assistant.
CONTEXT: Answer only from the retrieved policy context.
TASK: Respond to the customer's question accurately.
FORMAT: Return JSON with answer, sources, confidence.
LENGTH: Keep the answer concise.
NEGATIVE CONSTRAINT: Do not answer using information not present in the provided context.
FEW-SHOT EXAMPLE: Q: What are support hours? Context: in-app chat 24/7. A: Support is available via in-app chat 24/7.
'''


def classify_intent(
    s:State
):

    q=s['query'].lower()

    intent=(
        'policy_question'
        if any(
            k in q
            for k in KEYWORDS
        )
        else
        'general_question'
    )

    return {
        'intent':intent
    }


def retrieve_and_answer(
    s:State
):

    docs=retrieve(
        s['query'],
        3
    )

    top=docs[0]['text']

    if MOCK_LLM:

        ans=(
            'Based on the retrieved context: '
            f'{top[:200]}'
        )

    else:

        ans=real_llm_answer(
            s['query'],
            docs
        )

    return {
        'retrieved':docs,
        'answer':
            Answer(
                answer=ans,
                sources=[
                    d['id']
                    for d in docs
                ],
                confidence=1.0
            ).model_dump()
    }


def direct_answer(
    s:State
):

    if MOCK_LLM:

        ans=(
            'I can only answer questions '
            'about Zepto policies right now.'
        )

    else:

        ans=real_llm_answer(
            s['query'],
            []
        )

    return {
        'answer':
            Answer(
                answer=ans,
                sources=[],
                confidence=1.0
            ).model_dump()
    }


def real_llm_answer(
    query,
    docs
):
    raise RuntimeError(
        'MOCK_LLM=0 requires a configured '
        'LLM backend; mock mode is the graded baseline.'
    )
g=StateGraph(
    State
)

g.add_node(
    'classify_intent',
    classify_intent
)

g.add_node(
    'retrieve_and_answer',
    retrieve_and_answer
)

g.add_node(
    'direct_answer',
    direct_answer
)

g.add_edge(
    START,
    'classify_intent'
)

g.add_conditional_edges(
    'classify_intent',
    lambda s:s['intent'],
    {
        'policy_question':
            'retrieve_and_answer',

        'general_question':
            'direct_answer'
    }
)

g.add_edge(
    'retrieve_and_answer',
    END
)

g.add_edge(
    'direct_answer',
    END
)

graph=g.compile()


def ask(
    query
):

    return Answer(
        **graph.invoke(
            {
                'query':query
            }
        )['answer']
    )


if __name__=='__main__':

    for q in [
        'What is the delivery fee?',
        'What is the capital of France?'
    ]:

        print(
            json.dumps(
                ask(q).model_dump(),
                indent=2
            )
        )