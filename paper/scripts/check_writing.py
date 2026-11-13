#!/usr/bin/env python3
"""Writing checks: numbers in abstract/introduction, sentence-length distribution, vague wording."""
import re, glob, statistics as st

def plain(tex):
    t = tex.replace('\\%', ' percent ')
    t = re.sub(r'%.*', '', t)
    t = re.sub(r'\\begin\{(equation|align|algorithm|table\*?|figure\*?|tikzpicture|tabular|itemize|enumerate)\}.*?\\end\{\1\}', lambda m: ' ' if m.group(1) in ('equation','align','algorithm','table','table*','figure','figure*','tikzpicture','tabular') else m.group(0), t, flags=re.S)
    t = re.sub(r'\\(cite|ref|label|input|eqref)\{[^}]*\}', ' ', t)
    t = re.sub(r'\$[^$]*\$', ' X ', t)
    t = re.sub(r'\\(section|subsection|subsubsection)\*?\{[^}]*\}', ' ', t)
    t = re.sub(r'\\(textbf|emph|texttt|textit)\{([^}]*)\}', r'\2', t)
    t = re.sub(r'\\item(\[[^\]]*\])?', ' ', t)
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t)
    t = re.sub(r'[{}~\\]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

NUMWORDS = r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|hundred|thousand|million|first|second|third|fourth|fifth|sixth|once|twice|dozen)\b'
def numbers(text):
    digs = re.findall(r'(?<![A-Za-z\-])\d[\d.,]*%?', text)
    words = re.findall(NUMWORDS, text, flags=re.I)
    return digs, words

abstract = plain(open('sections/abstract.tex').read())
intro = plain(open('sections/01_introduction.tex').read())
for name, txt in (('ABSTRACT', abstract), ('INTRODUCTION', intro)):
    d, w = numbers(txt)
    print(f"{name}: digits {d} | number words {w}")

# sentence statistics over the body (sections 01-09, no appendix)
body = ' '.join(plain(open(f).read()) for f in sorted(glob.glob('sections/0*.tex')) + ['sections/abstract.tex'])
body = re.sub(r'(?<=\d)\.(?=\d)', '<d>', body)
body = re.sub(r'\b(e\.g|i\.e|et al|Fig|Eq|vs|No)\.', lambda m: m.group(0).replace('.', '<d>'), body)
sents = [s.replace('<d>', '.') for s in re.split(r'(?<=[.!?])\s+(?=[A-Z\\(])', body) if len(s.split()) >= 3]
L = [len(s.split()) for s in sents]
print(f"\nBODY: {len(L)} sentences | mean {st.mean(L):.1f} | median {st.median(L):.0f} | <9 words: {sum(l < 9 for l in L)} ({100*sum(l < 9 for l in L)/len(L):.1f}%) | >35 words: {sum(l > 35 for l in L)} ({100*sum(l > 35 for l in L)/len(L):.1f}%) | >40 words: {sum(l > 40 for l in L)}")
print("\nSentences with fewer than 9 words:")
for s in sents:
    if len(s.split()) < 9: print(f"  [{len(s.split())}] {s[:110]}")
print("\nSentences with more than 35 words:")
for s in sents:
    if len(s.split()) > 35: print(f"  [{len(s.split())}] {s[:140]}...")

# vague / informal wording
vague = ['big','huge','tiny','a lot','lots','stuff','thing','things','nice','good','bad','great','very','really','quite','pretty','kind of','sort of','a bit','maybe','obviously','clearly','simply','just ','easy','hard ','cheap','serious','many ','several','various','numerous','important','interesting','robust','leverage','comprehensive','crucial','delve','landscape','novel','state-of-the-art','significant','strong ','weak ','small ','large ','huge','fast ','slow','get ','got ','can\'t','don\'t','won\'t','doesn\'t','isn\'t','it\'s','we\'ll','in order to','a number of','plays a']
low = body.lower()
print("\nVague/informal wording found:")
for v in vague:
    n = len(re.findall(r'\b' + re.escape(v.strip()) + (r'\b' if v.endswith(' ') or v.strip()==v else ''), low)) if v.strip().isalpha() else low.count(v)
    if n: 
        ctx = [body[max(0, m.start()-35):m.end()+35] for m in re.finditer(re.escape(v.strip()), body, flags=re.I)][:2]
        print(f"  {v.strip()!r}: {n}   e.g. ...{ctx[0]}...")
