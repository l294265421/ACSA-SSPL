"""
https://spacy.io/universe/project/self-attentive-parser
https://github.com/nikitakit/self-attentive-parser
https://parser.kitaev.io/
"""
import spacy
import pysbd
from pysbd.utils import PySBDFactory
from benepar.spacy_plugin import BeneparComponent
import matplotlib.pyplot as plt
import networkx as nx


nlp = spacy.load('en_core_web_sm')
doc = nlp('Frybread is good, but there isn\'t enough substance.')
words = [word.text for word in doc]

g = nx.DiGraph()
g.add_nodes_from(words)

g.add_edges_from([(words[0], words[1])])

nx.draw(g, with_labels=True)
plt.show()

print()

# nlp.add_pipe(BeneparComponent('benepar_en'))
# doc = nlp('Frybread is good, but there isn\'t enough substance.')

# sent = list(doc.sents)[0]
# print(sent._.parse_string)
# # (S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
# print(sent._.labels)
# # ('S',)
# print(list(sent._.children)[0])
#
# constituents = list(sent._.constituents)
# print(constituents)