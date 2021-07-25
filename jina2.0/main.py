import json
import os
import uuid
import jina.types.document.generators
import numpy as np
from jina import Document, DocumentArray, Executor, Flow, requests
from scipy.spatial import distance

tutorual_dataset = './simple_dataset.txt'
EXAMPLE_DATASET = './data/translated_resumes_example.json'


class CharEmbed(Executor):  # a simple character embedding with mean-pooling
    offset = 32  # letter `a`
    dim = 127 - offset + 1  # last pos reserved for `UNK`
    char_embd = np.eye(dim) * 1  # one-hot embedding for all chars

    @requests
    def foo(self, docs: DocumentArray, **kwargs):

        for d in docs:
            print(d)
            if d.chunks:
                for chunk in d.chunks:
                    print(chunk.text)
                    r_emb = [ord(c) - self.offset if self.offset <= ord(c) <= 127 else (self.dim - 1) for c in
                             chunk.text]
                    # average pooling
                    chunk.embedding = self.char_embd[r_emb, :].mean(axis=0)
            else:
                r_emb = [ord(c) - self.offset if self.offset <= ord(c) <= 127 else (self.dim - 1) for c in d.text]
                # average pooling
                d.embedding = self.char_embd[r_emb, :].mean(axis=0)


def get_distances(query, chunks, metric='euclidean'):
    dists = []
    m = getattr(distance, metric)
    for chunk in chunks:
        e = np.stack(chunk[2])
        d = m(query, e)
        dists.append((chunk[0], chunk[1], d))
    return dists


class Indexer(Executor):
    _docs = DocumentArray()  # for storing all documents in memory

    @requests(on='/index')
    def foo(self, docs: DocumentArray, **kwargs):
        self._docs.extend(docs)  # extend stored `docs`

    @requests(on='/search')
    def bar(self, docs: DocumentArray, **kwargs):
        q = np.stack(docs.get_attributes('embedding'))  # get all embeddings from query docs
        chunks = self._docs.get_attributes('chunks')
        cc = [chunk.get_attributes('embedding') for chunk in chunks]
        v = []
        for doc_id, c in enumerate(cc):
            for chunk_id, ccc in enumerate(c):
                v.append((doc_id, chunk_id, ccc))

        distances = get_distances(q[0], v, )
        distances_sorted = [dist for dist in sorted(distances, key=lambda dist: dist[2])]
        docs_matched = {ds[0]: ds[2] for ds in distances_sorted[-1::-1]}

        docs[0].matches = [Document(self._docs[int(doc_idx)], copy=True, scores={'euclid': doc_dist})
                           for doc_idx, doc_dist in docs_matched.items()]
        docs[0].matches.sort(key=lambda m: m.scores['euclid'].value)  # sort matches by their values



f = Flow(port_expose=12345, protocol='http', cors=True).add(uses=CharEmbed, parallel=2).add(
    uses=Indexer)  # build a Flow, with 2 parallel CharEmbed, tho unnecessary
with f:
    with open(EXAMPLE_DATASET) as resumes_json:
        documents = json.load(resumes_json)['resumes']
        for i, doc in enumerate(documents):
            doc['id'] = i

    # documents_array = DocumentArray()
    # for doc in documents:
    #     jina_doc = Document(
    #         id=doc['id'],
    #         document= doc,
    #         # field_resolver={'last_name': 'last_name'},
    #         content=doc['position'],
    #         chunks=[Document(id=i, text=we['work_description']) for i, we in enumerate(doc['full_work_experience'])]
    #     )
    #     documents_array.append(jina_doc)
    def gen():

        for  l in os.listdir('./data/resumes/'):

            with open('./data/resumes/'+l) as j:
                data = json.load(j)
                data['id'] =uuid.uuid4().hex


                s = json.dumps(data)

                yield s
    g = gen()

    documents_array  = jina.types.document.generators.from_ndjson(g, field_resolver={
        'location':'ll',
        'position':'text',
    })
    documents_array = list(documents_array)
    for jdoc in documents_array:
        jdoc.chunks =[Document(id=i, text=we['work_description']) for i, we in enumerate(jdoc.tags['full_work_experience'])]
        print('he')
    f.post('/index', documents_array)  # index all lines of _this_ file
    f.block()  # block for listening request
