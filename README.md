# Advent of Code (Golf) 2024

Rules:

1. One single-line Python expression
2. Print a tuple `(answer1, answer2)`

## 01

```bash
$ python3 -c 'print((l:=tuple(map(sorted,zip(*(map(int,l.strip().split())for l in __import__("sys").stdin)))),sum(abs(a-b)for a,b in zip(*l)),sum(a*sum(1 for x in l[1] if x==a)for a in l[0]))[1:])' < example
(11, 31)
```

Started doing this in Go, but got the feeling that "this would be a one-liner in Python", and here we are.

## 02

```bash
$ python3 -c 'print(((l:=[[int(x)for x in l.strip().split()]for l in __import__("sys").stdin]),tuple(sum([any([all([1<=b-a<=3 for a,b in list(zip(l,l[1:]))])for l in r])for r in[[l[:i]+l[i+d:]for i,_ in enumerate(l)]for l in[list(reversed(x))if x[0]>x[-1] else x for x in l]]])for d in(0,1)))[1])' < example
(2, 4)
```

`zip(lst, lst[1:])` is a nice pattern for a sliding window thingy.

## 03

```bash
$ python3 -c 'print(((x:="do()"+"".join(__import__("sys").stdin),f:=lambda d:sum(int(x[0])*int(x[1])for x in[x.split(",")for x in"".join((x.split("do()",maxsplit=1)+[""])[1]for x in x.split(d)).split("mul(")for x in x.split(")")]if len(x)==2 and all(x.isnumeric()for x in x))),f("🐍"),f("don'\''t()"))[-2:])' < example
(161, 48)
```

Split on `don't()` (or `🐍` in Part One 😉) and drop everything before the first `do()` in each segment.

## 04

```bash
$ python3 -c 'print((x:=[x.strip()for x in __import__("sys").stdin],sum(x=="XMAS"for x in["".join(x[r+a*i][c+b*i]for i in range(4))for r,_ in enumerate(x)for c,_ in enumerate(x[0])for a,b in [(a,b)for a in(-1,0,1)for b in(-1,0,1)if a or b]if 0<=r+a*3<len(x)and 0<=c+b*3<len(x[0])]),len([x for x in [sum(x)for x in [["".join((x[r-a][c-b],x[r][c],x[r+a][c+b]))=="MAS"for a,b in[(a,b)for a in(-1,1)for b in(-1,1)]if 0<=r-a<len(x)and 0<=r+a<len(x)and 0<=c-b<len(x[0])and 0<=c+b<len(x[0])]for r,_ in enumerate(x)for c,_ in enumerate(x[0])]]if x==2]))[1:])' < example
(18, 9)
```

Indices and indexes...

## 05

```bash
$ python3 -c 'print(globals().__setitem__("a",lambda k,v:globals().__setitem__(k,v))or a("l",[l.strip()for l in __import__("sys").stdin])or a("r",[r.split("|")for r in l[:l.index("")]])or a("p",[u.split(",")for u in l[l.index("")+1:]])or a("d",lambda g,u:u in V or V.add(u)or([d(g,v)for v in g.get(u,[])]or 1)and O.append(u))or a("t",lambda g:a("V",set())or a("O",[])or ([d(g,n)for n in g]or 1)and O[::-1])or a("G",{k:[v for u,v in r if u==k]for k,_ in r})or a("c",lambda G,O:a("P",{n:i for i,n in enumerate(O)})or all(u not in P or v not in P or P[u]<P[v]for u in G for v in G[u]))or(sum(int(u[len(u)//2])for u in p if c(G,u)),sum(int(t({k:[x for x in v if x in u]for k,v in G.items()if k in u})[len(u)//2])for u in p if not c(G,u))))' < example
(143, 123)
```

No walrus! This little utility for assigning variables was kinda nice `globals().__setitem__("a",lambda k,v:globals().__setitem__(k,v))`. DFS-based topological sort.

## 06

```bash
$ python3 -c 'print((L:=[list(x.strip())for x in open(0).readlines()],I:=next((i,j)for i,r in enumerate(L) for j,v in enumerate(r)if v=="^"),F:=lambda o=0:(l:=[r[:]for r in L],o and(l[o[0]].__setitem__(o[1],"#")),l[I[0]].__setitem__(I[1],"X"),D:=iter(lambda x=[(0,-1),(-1,0),(0,1),(1,0)]:x.append(x.pop(0))or x[0],1),d:=next(D),S:=set(),s:=[I[0],I[1],d[0],d[1]],f:=lambda:((n:=[s[0],s[1],s[2],s[3]],s.clear(),s.extend(n),t:=tuple(s),a:=t in S,S.add(t)if not a else a,b:=l[s[0]+s[2]][s[1]+s[3]]=="#",(d:=next(D),s.__setitem__(2,d[0]),s.__setitem__(3,d[1]))if not a and b else a,(s.__setitem__(0,s[0]+s[2]),s.__setitem__(1,s[1]+s[3]),l[s[0]].__setitem__(s[1],"X"))if not a and not b else a)and a),(R:=[False],list(iter(lambda x=[False]:(x.pop()or x.append((not(0<s[0]<len(l)-1 and 0<s[1]<len(l[0])-1)and(R.pop()or R.append(sum(v=="X"for r in l for v in r))or False))or(bool(R[0])or R.pop()or R.append(f())or bool(R[0])))or x[0]),True))))and(R[0],l),a:=F(),b:=sum(x for x,_ in[F((i,j))for i,r in enumerate(a[1])for j,v in enumerate(r)if v=="X"and(i,j)!=I]if isinstance(x,bool)),a[0],b)[-2:])' < example
(41, 6)
```

Okay, today was really weird. But I learned that you can construct something like a `while` loop with `iter(callable, sentinel)`, which was kinda cool:

```python
>>> list(iter(lambda l=[0]: l.append((x := l.pop()) + 1) or x, 10))
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

And also that you can read from stdin with just `open(0)`.

## 07

```bash
$ python3 -c 'print((I:=[(int(x[0]),[int(x)for x in x[1].split()])for x in[x.split(":")for x in open(0).read().splitlines()]],P:=[[i for i,n in enumerate(N[1:],start=1)]for t,N in I],(f:=lambda N,C,s:[s.append((s.pop()+N[i])if o=="+"else((s.pop()*N[i])if o=="*"else(int(str(s.pop())+str(N[i])))))or s[0]for i,o in C][-1]),C:=lambda O:((list(c)for c in __import__("itertools").product(*[[(x,o)for o in O]for x in p]))for p in P),E:=([(t,N,[(0,"+")]+c)for c in c]for(t,N),c in zip(I,C(["+","*"]))),R:=[any(f(N,C,[0])==t for t,N,C in e)for e in E],a:=sum(t for t,r in zip([t for t,_ in I],R)if r),E:=([(t,N,[(0,"+")]+c)for c in c]for(t,N),c in zip(I,C(["+","*","|"]))),R:=(any(f(N,C,[0])==t for t,N,C in e)for e in E),b:=sum(t for t,r in zip([t for t,_ in I],R)if r),a,b)[-2:])' < example
(3749, 11387)
```

Today I tried solving the problem in a tuple directly, instead of trying to solve it the sane way first and then converting it to a single expression. This approach might be easier overall actually, will try it again tomorrow.

Using `any(generator expression)` is nice since it short-circuits when it finds a `True`. Also, list comprehensions allocates a lot of memory, so for large search spaces generator expressions can be more appropriate.

## 08

```bash
$ python3 -c 'print((print:=lambda*_:None,l:=open(0).read().splitlines(),print("\n".join(l)),F:={f for r in l for f in r if f!="."},print(F),A:={f:[(i,j)for i,r in enumerate(l)for j,v in enumerate(r)if v==f]for f in F},print(A),N:={f:[(p1[0]*2 - p2[0],p1[1]*2 - p2[1])for p1 in A[f]for p2 in A[f]if p1!=p2]for f in A},print(N),n:={(i,j)for n in N.values()for(i,j)in n if 0 <=i < len(l)and 0 <=j < len(l[0])},print(n),print("\n".join("".join(l[i][j]if(i,j)not in n else"#"for j,_ in enumerate(r))for i,r in enumerate(l))),print(O:=len(n)),D:={f:[(p1,(p2[0]- p1[0],p2[1]- p1[1]))for p1,p2 in __import__("itertools").combinations(A[f],2)]for f in A},print(D),d:=[x for d in D.values()for x in d],print(d),g:=lambda s,d:s.append((s[-1][0]+ d[0],s[-1][1]+ d[1]))or(0 <=s[-1][0]< len(l)and 0 <=s[-1][1]< len(l[0])),R:=[(s:=[a])and list(iter(__import__("functools").partial(g,s,b),False))and s for(a,b)in d],R:=(R +[(s:=[a])and list(iter(__import__("functools").partial(g,s,(-b[0],-b[1])),False))and s for(a,b)in d]),n:={(i,j)for r in R for(i,j)in r if 0 <=i < len(l)and 0 <=j < len(l[0])},print(n),print("\n".join("".join(l[i][j]if(i,j)not in n else"#"for j,_ in enumerate(r))for i,r in enumerate(l))),print(T:=len(n)),O,T)[-2:])' < example
(14, 34)
```

Today I took an even more structured approach:

```python3
print((
# Remove to get debug output
print := lambda *_: None,

# Read lines from stdin
l := open(0).read().splitlines(),
print("\n".join(l)),

# Build a set of all frequencies
F := {f for r in l for f in r if f != "."},
print(F),

# Build a dict of antennas {"frequency": [positions]}
A := {f: [(i,j) for i, r in enumerate(l) for j, v in enumerate(r) if v == f] for f in F},
print(A),

# For each frequency, go through each pair of antennas and find their antinode
# Let (p1, p2) create the first difference vector (p1 - p2), meaning the antinode ends up in (p1*2 - p2)
# Then (p2, p1) creates the other
# Antinode position is (p1 + (p1 - p2)) and vice versa
N := {f: [(p1[0]*2 - p2[0], p1[1]*2 - p2[1]) for p1 in A[f] for p2 in A[f] if p1 != p2] for f in A},
print(N),

# Flatten the antinodes to a set and filter out-of-bounds
n := {(i, j) for n in N.values() for (i, j) in n if 0 <= i < len(l) and 0 <= j < len(l[0])},
print(n),

# Print the map, looks good
print("\n".join("".join(l[i][j] if (i, j) not in n else "#" for j, _ in enumerate(r)) for i, r in enumerate(l))),

# Answer to Part One
print(O := len(n)),

# All right, time for Part Two
# Let's grab the code above and make tuples (start, difference vector) instead
D := {f: [(p1, (p2[0] - p1[0], p2[1] - p1[1])) for p1, p2 in __import__("itertools").combinations(A[f], 2)] for f in A},
print(D),

# Flatten D to only tuples-of-tuples
d := [x for d in D.values() for x in d],
print(d),

# Ok, so now we do a weird while loop I guess
# Start from start and add difference vector until we go out-of-bounds
# Then subtract until we go out-of-bounds
# Using a list of the state makes some sense at least

# Since append returns None we can do "or condition", we then do iter(callable, sentinel=False)
# to get our while-loop-like behavior, breaking when we are out-of-bounds
g := lambda s, d: s.append((s[-1][0] + d[0], s[-1][1] + d[1])) or (0 <= s[-1][0] < len(l) and 0 <= s[-1][1] < len(l[0])),

# We can use "and" to control what value we get out from the comprehension, start with adding
R := [(s := [a]) and list(iter(__import__("functools").partial(g, s, b), False)) and s for (a, b) in d],

# And now we do the subtracting
R := (R + [(s := [a]) and list(iter(__import__("functools").partial(g, s, (-b[0], -b[1])), False)) and s for (a, b) in d]),

# Make it a set and we should have our antinodes
n := {(i, j) for r in R for (i, j) in r if 0 <= i < len(l) and 0 <= j < len(l[0])},
print(n),

# Print the map, looks good!
print("\n".join("".join(l[i][j] if (i, j) not in n else "#" for j, _ in enumerate(r)) for i, r in enumerate(l))),

# Answer to Part Two
print(T := len(n)),

# The rules are to print (Part One, Part Two)
O, T)[-2:])
```

Then I minified it with:

```python
import sys

c = "()[],:=!*\""

x = "\n".join(x for x in open(sys.argv[1]).read().splitlines() if not x.startswith("#")).split()

s = ""
for a, b in zip(x, x[1:]):
    s += a
    if a[-1] not in c and b[0] not in c:
        s += " "
s += x[-1]

print(s)
```

## 09

```bash
$ python3 -c 'print((X:=open(0).read().strip(),I:=__import__("itertools"),F:=[int(x)for x in X[::2]],S:=[int(x)for x in X[1::2]],b:=lambda i,f,s:([i]*f,([]if s is None else["."]*s)),B:=[y for x in(y for x in(b(i,f,s)for i,(f,s)in enumerate(I.zip_longest(F,S)))for y in x)for y in x],l:=[0],r:=[len(B)-1],m:=lambda:l[0]<r[0]and B[l[0]]!="."and l.append(l.pop()+1)and True,M:=lambda:list(iter(m,False))and False or l[0]<r[0],M(),n:=lambda:l[0]<r[0]and B[r[0]]=="."and r.append(r.pop()-1)and True,N:=lambda:list(iter(n,False))and False or l[0]<r[0],N(),f:=lambda:M()and N()and(B.__setitem__(l[0],B[r[0]])or B.__setitem__(r[0],".")or True),list(iter(f,False)),O:=sum(i*l for i,l in enumerate(x for x in B if x!=".")),B:=[y for x in(y for x in(b(i,f,s)for i,(f,s)in enumerate(I.zip_longest(F,S)))for y in x)for y in x],B:=[(t,len(list(x)))for t,x in I.groupby(B)],R:=list(I.accumulate(l for _,l in B)),R:=([0]+R)[:-1],K:=[(i,t,l)for(t,l),i in zip(B,R)],F:=[(i,t,l)for i,t,l in K if t!="."],D:=F[:1],F:=F[1:],S:=[(i,l)for i,t,l in K if t=="."],f:=lambda:(x:=F.pop())and(s:=next(((i,s)for i,s in enumerate(S)if s[1]>=x[2]and s[0]<x[0]),None))and(S.pop(s[0]))and(s[1][1]-x[2])and S.insert(s[0],(s[1][0]+x[2],s[1][1]-x[2]))or D.append((x[0],x[1],x[2])if s is None else(s[1][0],x[1],x[2])),g:=lambda:f()and False or bool(F),list(iter(g,False)),D.sort(),T:=sum(sum(x*t for x in range(i,i+l))for i,t,l in D),O,T)[-2:])' < example
(1928, 2858)
```

Ok, I think something clicked yesterday. Think "Haskell", walrus is `let`, then sprinkle some weird list manipulation on top. Unminified:

```python
print((
# Read from stdin
X := open(0).read().strip(),
print(X),

I := __import__("itertools"),

# Split to files F and free space S
F := [int(x) for x in X[::2]],
print(F),
S := [int(x) for x in X[1::2]],
print(S),

# Build blocks B
b := lambda i, f, s: ([i] * f, ([] if s is None else ["."] * s)),
B := [y for x in (y for x in (b(i, f, s) for i, (f, s) in enumerate(I.zip_longest(F, S))) for y in x) for y in x],
print("".join(map(str, B))),

# Keep pointers from left and right
# Loop invariant: l is the first "." and r is the last number
# Move value from r to l, then update pointers
l := [0],
r := [len(B)-1],

# Returns True if we should keep moving l
m := lambda: l[0] < r[0] and B[l[0]] != "." and l.append(l.pop() + 1) and True,
M := lambda: list(iter(m, False)) and False or l[0] < r[0],
M(),
print("l:", l),

# Returns True if we should keep moving r
n := lambda: l[0] < r[0] and B[r[0]] == "." and r.append(r.pop() - 1) and True,
N := lambda: list(iter(n, False)) and False or l[0] < r[0],
N(),
print("r:", r),

# Do work until done
f := lambda: M() and N() and (B.__setitem__(l[0], B[r[0]]) or B.__setitem__(r[0], ".") or True),
list(iter(f, False)),

# Get answer for Part One
O := sum(i * l for i, l in enumerate(x for x in B if x != ".")),
print("Part One:", O),

# Reset B
B := [y for x in (y for x in (b(i, f, s) for i, (f, s) in enumerate(I.zip_longest(F, S))) for y in x) for y in x],
print(B),

# Construct list F for "number groups": (index, type, length)
# Construct list S for spaces: (index, length)
# D is a list of done number groups: (index, type, length)
B := [(t, len(list(x))) for t, x in I.groupby(B)],
print(B),
R := list(I.accumulate(l for _, l in B)),
R := ([0] + R)[:-1],
print(R),
K := [(i, t, l) for (t, l), i in zip(B, R)],
F := [(i, t, l) for i, t, l in K if t != "."],
D := F[:1],
F := F[1:],
S := [(i, l) for i, t, l in K if t == "."],
print(F),
print(D),
print(S),
print(),

# Pop from F, find first space where number fits
f := lambda: (x := F.pop()) and (s := next(((i, s) for i, s in enumerate(S) if s[1] >= x[2] and s[0] < x[0]), None)) and (S.pop(s[0])) and (s[1][1] - x[2]) and S.insert(s[0], (s[1][0] + x[2], s[1][1] - x[2])) or D.append((x[0], x[1], x[2]) if s is None else (s[1][0], x[1], x[2])),

# Do work
g := lambda: f() and False or bool(F),
list(iter(g, False)),
D.sort(),
print(D),

# Get answer for Part Two
T := sum(sum(x*t for x in range(i, i+l)) for i, t, l in D),

# Print (Part One, Part Two)
O, T)[-2:])
```

## 10

```bash
$ python3 -c 'print((M:=[[int(x)for x in m]for m in open(0).read().splitlines()],S:=[(i,j)for i,m in enumerate(M)for j,x in enumerate(m)if x==0],N:=lambda v:[(v[0]+a,v[1]+b)for a,b in[(0,1),(1,0),(0,-1),(-1,0)]if 0<=v[0]+a<len(M)and 0<=v[1]+b<len(M[0])and M[v[0]+a][v[1]+b]==M[v[0]][v[1]]+1],D:=lambda v,V,F:V.add(v)or[(F.append(n)if M[n[0]][n[1]]==9 else None)or D(n,V,F)for n in N(v)if n not in V],R:=[D(s,V:=set(),F:=[])and False or(s,F)for s in S],O:=sum(len(f)for _,f in R),D:=lambda v,V,p,P:V.add(v)or p.append(v)or(P.append(list(p))if M[v[0]][v[1]]==9 else None)or[D(n,V,p,P)for n in N(v)if n not in V]and False or V.remove(v)or p.pop(),R:=[D(s,V:=set(),[],P:=[])and False or(s,P)for s in S],T:=sum(len(f)for _,f in R),O,T)[-2:])' < example
(36, 81)
```

Unminified:

```python
print((
# Read map from stdin
M := [[int(x) for x in m] for m in open(0).read().splitlines()],
print(f"{M=}"),

# Find all start coordinates 0
S := [(i,j) for i,m in enumerate(M) for j,x in enumerate(m) if x==0],
print(f"{S=}"),

# Generate v's edges
N := lambda v: [(v[0]+a,v[1]+b) for a,b in [(0,1), (1,0), (0,-1), (-1,0)] if 0<=v[0]+a<len(M) and 0<=v[1]+b<len(M[0]) and M[v[0]+a][v[1]+b] == M[v[0]][v[1]]+1],
print(f"{N(S[0])=}"),

# DFS current vertex, visited, finish 9's
D := lambda v, V, F: V.add(v) or [(F.append(n) if M[n[0]][n[1]]==9 else None) or D(n, V, F) for n in N(v) if n not in V],

# Do work
R := [D(s, V:=set(), F:=[]) and False or (s, F) for s in S],
print(f"{R=}"),

# Part One
O := sum(len(f) for _,f in R),

# Let's keep track of paths instead and let P be all paths
D := lambda v, V, p, P: V.add(v) or p.append(v) or (P.append(list(p)) if M[v[0]][v[1]]==9 else None) or [D(n, V, p, P) for n in N(v) if n not in V] and False or V.remove(v) or p.pop(),

# Do work
R := [D(s, V:=set(), [], P:=[]) and False or (s, P) for s in S],
print(f"{R=}"),

# Part Two
T := sum(len(f) for _,f in R),

O, T)[-2:])
```

## 11

```bash
$ python3 -c 'print((X:=[int(x)for x in open(0).read().split()],X:={k:len([x for x in X if x==k])for k in set(X)},I:=lambda N,i,n:N.__setitem__(i,N[i]+n),f:=lambda P:((N:=__import__("collections").defaultdict(int))or[(s:=str(x))and(n:=P[x])and False or(I(N,1,n)if x==0 else I(N,int(s[:len(s)//2]),n)or I(N,int(s[len(s)//2:]),n)if len(s)% 2==0 else I(N,x*2024,n))for x in P])and False or N,P:=X,[(P:=f(P))for _ in range(25)],O:=sum(P.values()),P:=X,[(P:=f(P))for _ in range(75)],T:=sum(P.values()),O,T)[-2:])' < example
(55312, 65601038650482)
```

Unminified:

```python
print((
# Read from stdin and build dict {number: count}
X := [int(x) for x in open(0).read().split()],
X := {k: len([x for x in X if x == k]) for k in set(X)},
print(X),

# Helper function to do: N[i] += n
I := lambda N, i, n: N.__setitem__(i, N[i]+n),

# Actual logic
f := lambda P: (
# Create new dict N
(N := __import__("collections").defaultdict(int))
or [(s := str(x)) and (n := P[x]) and False
# 0 turns into 1
or (I(N,1,n) if x == 0
# Split if even number of digits
else I(N,int(s[:len(s)//2]),n) or I(N,int(s[len(s)//2:]),n) if len(s) % 2 == 0
# Otherwise multiply by 2024
else I(N,x*2024,n)) for x in P])
# Return N
and False or N,

# Part One
P := X,
[(P := f(P)) for _ in range(25)],
O := sum(P.values()),

# Part Two
P := X,
[(P := f(P)) for _ in range(75)],
T := sum(P.values()),

O, T)[-2:])
```

## 12

```bash
$ python3 -c 'print((M:=[list(x)for x in open(0).read().splitlines()],e:=enumerate,L:=__import__("collections").defaultdict(list),[L[l].append((i,j))for i,m in e(M)for j,l in e(m)],F:=__import__("functools"),D:=[(0,1),(-1,0),(0,-1),(1,0)],b:=lambda M,Q,V:(v:=Q.pop(),V.add(v),Q.extend([(i,j)for i,j in((v[0]+a,v[1]+b)for a,b in D)if(i,j)not in V and 0<=i<len(M)and 0<=j<len(M[0])and M[i][j]==M[v[0]][v[1]]]),bool(Q))[-1],B:=lambda M,s:(Q:=[s],V:=set(),list(iter(F.partial(b,M,Q,V),False)),V)[-1],c:=lambda U,C:(s:=U.pop(),V:=B(M,s),C.append(V),U.difference_update(V),bool(U))[-1],R:=[(U:=set(X),C:=[],list(iter(F.partial(c,U,C),False)),C)[-1]for X in L.values()],R:=[C for r in R for C in r],P:=[sum(M[i][j]!=M[i+a][j+b]if 0<=i+a<len(M)and 0<=j+b<len(M[0])else 1 for i,j in C for a,b in D)for C in R],O:=sum(len(C)*p for C,p in zip(R,P)),D:=[(-.5,.5),(.5,.5),(.5,-.5),(-.5,-.5)],X:=[{(i,j):[(i+a,j+b)for a,b in D]for i,j in C} for C in R],Y:=[[(v,k)for k,V in x.items()for v in V]for x in X],Z:=[{k:[v for l,v in y if k==l]for k,v in y} for y in Y],C:=[{k:v for k,v in z.items()if len(v)%2==1} for z in Z],C:=[len(c)for c in C],D:=[{k:v for k,v in z.items()if len(v)==2 and abs(v[0][0]-v[1][0])==abs(v[0][1]-v[1][1])==1} for z in Z],D:=[sum(map(len,d.values()))for d in D],S:=[c+d for c,d in zip(C,D)],T:=sum(len(C)*s for C,s in zip(R,S)),O,T)[-2:])' < example
(1930, 1206)
```

Part Two was painful. Unminified:

```python
print((
# Read map from stdin
M:=[list(x) for x in open(0).read().splitlines()],
print(f"{M=}"),

# Dict {letter: [(i,j)]}
e:=enumerate,
L:=__import__("collections").defaultdict(list),
[L[l].append((i,j)) for i,m in e(M) for j,l in e(m)],

# BFS
F:=__import__("functools"),
D:=[(0,1),(-1,0),(0,-1),(1,0)],
# Inner loop
b:=lambda M, Q, V: (
# Get next vertex
v:=Q.pop(),
# Mark v as visited
V.add(v),
# Add all neighbors to queue
Q.extend([(i,j) for i,j in ((v[0]+a,v[1]+b) for a,b in D) if (i,j) not in V and 0<=i<len(M) and 0<=j<len(M[0]) and M[i][j]==M[v[0]][v[1]]]),
bool(Q))[-1],
# Outer loop
B:=lambda M, s: (
Q:=[s],
V:=set(),
# While Q is not empty
list(iter(F.partial(b,M,Q,V), False)),
V)[-1],

# Collect cliques
# For letter:
# While U:
# pop s, get clique V = BFS(s), U -= V
c:=lambda U, C: (
s:=U.pop(),
V:=B(M, s),
C.append(V),
U.difference_update(V),
bool(U))[-1],
R:=[(U:=set(X), C:=[], list(iter(F.partial(c,U,C),False)), C)[-1] for X in L.values()],
# Flatten to list of cliques
R:=[C for r in R for C in r],
print(f"{R=}"),

# An edge contributes to the perimiter if it connects to another letter or the outer border
P:=[sum(M[i][j]!=M[i+a][j+b] if 0<=i+a<len(M) and 0<=j+b<len(M[0]) else 1 for i,j in C for a,b in D) for C in R],
print(f"{P=}"),

# Part One = area * perimeter
O:=sum(len(C)*p for C,p in zip(R,P)),

# All right, Part Two should be something like counting clique shape corners instead
# For each clique/shape:
# Each cell has 4 candidate corners
# If it's not shared by any other cells, it's a regular "outwards corner"
# If 3 cells share a corner, it's an "inwards corner"

# There's also the weird case between the two B shapes:
# AAAAAA
# AAABBA
# AAABBA
# ABBAAA
# ABBAAA
# AAAAAA

# Specifically:
# AB
# BA

# So if exactly 2 cells have a corner in common, and the cells are diagonally adjacent,
# they form a corner in each direction as well
D := [(-.5,.5),(.5,.5),(.5,-.5),(-.5,-.5)],
X:=[{(i,j): [(i+a,j+b) for a,b in D] for i,j in C} for C in R],
print(f"{X=}"),
Y:=[[(v,k) for k,V in x.items() for v in V] for x in X],
print(f"{Y=}"),
Z:=[{k:[v for l,v in y if k==l] for k,v in y} for y in Y],
print(f"{Z=}"),

# Outwards/inwards corners
C:=[{k:v for k,v in z.items() if len(v)%2==1} for z in Z],
print(f"{C=}"),
C:=[len(c) for c in C],

# Weird diagonals
D:=[{k:v for k,v in z.items() if len(v)==2 and abs(v[0][0]-v[1][0])==abs(v[0][1]-v[1][1])==1} for z in Z],
print(f"{D=}"),
D:=[sum(map(len,d.values())) for d in D],

# Sum all corners
S:=[c+d for c,d in zip(C,D)],
print(S),

# Part Two = area * corners
T:=sum(len(C)*s for C,s in zip(R,S)),

O, T)[-2:])
```

## 13

```bash
$ python3 -c 'print((X:=[x for x in open(0).read().splitlines()if x],X:=[[tuple(map(int,__import__("re").findall(r"\d+",x)))for x in X[i*3:i*3+3]]for i in range(len(X)//3)],F:=__import__("fractions").Fraction,M:=[F(p[1]-F(p[0]*b[1],b[0]),a[1]-F(a[0]*b[1],b[0]))for a,b,p in X],N:=[F(p[0]-m*a[0],b[0])for(a,b,p),m in zip(X,M)],Z:=[(int(m),int(n))for m,n in zip(M,N)if m.is_integer()and n.is_integer()and 0<=int(m)<=100 and 0<=int(n)<=100],T:=[m*3+n for m,n in Z],O:=sum(T),d:=10000000000000,X:=[(a,b,(p[0]+d,p[1]+d))for a,b,p in X],M:=[F(p[1]-F(p[0]*b[1],b[0]),a[1]-F(a[0]*b[1],b[0]))for a,b,p in X],N:=[F(p[0]-m*a[0],b[0])for(a,b,p),m in zip(X,M)],Z:=[(int(m),int(n))for m,n in zip(M,N)if m.is_integer()and n.is_integer()and 0<=int(m)and 0<=int(n)],T:=[m*3+n for m,n in Z],T:=sum(T),O,T)[-2:])' < example
(480, 875318608908)
```

`fractions` to the rescue! Unminified:

```python
print((
# Parse each claw machine
X:=[x for x in open(0).read().splitlines() if x],
X:=[[tuple(map(int, __import__("re").findall(r"\d+", x))) for x in X[i*3:i*3+3]] for i in range(len(X)//3)],
print(f"{X=}"),

# X is [[(ax, ay), (bx, by), (px, py)], ...]
# Find integers m, n such that:
# (1) m*ax + n*bx = px
# (2) m*ay + n*by = py
# (3) 0 <= m <= 100
# (4) 0 <= n <= 100
# (5) minimize t = m*3 + n (there can only be one solution)

# (1) m*ax + n*bx = px
# <=> (6) n = (px - m*ax)/bx
# insert in (2) m*ay + (px - m*ax)*by/bx = py
# <=> m*ay + px*by/bx - m*ax*by/bx = py
# <=> m(ay-ax*by/bx) = py - px*by/bx
# <=> (7) m = (py - px*by/bx)/(ay-ax*by/bx)

F:=__import__("fractions").Fraction,

# Using (7) to get floats m
M:=[F(p[1]-F(p[0]*b[1],b[0]),a[1]-F(a[0]*b[1],b[0])) for a,b,p in X],
print(f"{M=}"),

# Using (6) to get floats n
N:=[F(p[0]-m*a[0],b[0]) for (a,b,p),m in zip(X,M)],
print(f"{N=}"),

print(f"{list(zip(M,N))=}"),

# integers 0 <= m, n <= 100
Z:=[(int(m), int(n)) for m,n in zip(M,N) if m.is_integer() and n.is_integer() and 0<=int(m)<=100 and 0<=int(n)<=100],
print(f"{Z=}"),

# Part One: sum tokens
T:=[m*3+n for m,n in Z],
print(f"{T=}"),
O:=sum(T),

# Nice, I kind of had the feeling that simulation wasn't the right approach ;)
d:=10000000000000,
X:=[(a,b,(p[0]+d,p[1]+d)) for a,b,p in X],
M:=[F(p[1]-F(p[0]*b[1],b[0]),a[1]-F(a[0]*b[1],b[0])) for a,b,p in X],
N:=[F(p[0]-m*a[0],b[0]) for (a,b,p),m in zip(X,M)],
Z:=[(int(m), int(n)) for m,n in zip(M,N) if m.is_integer() and n.is_integer() and 0<=int(m) and 0<=int(n)],
print(Z),

# Part Two: sum tokens
T:=[m*3+n for m,n in Z],
T:=sum(T),

O, T)[-2:])
```

## 14

```bash
$ python3 -c 'print((X:=open(0).read().splitlines(),X:=[(a:=x[2:].split(","))and(b:=a[1].split("v="))and(int(a[0]),int(b[0]),int(b[1]),int(a[2]))for x in X],S:=(11,7)if len(X)==12 else(101,103),R:=list(X),P:=lambda R:(M:=[[0 for _ in range(S[0])]for _ in range(S[1])],[M[q].__setitem__(p,M[q][p]+1)for p,q,v,w in R],[[print(c if c else".",end="")for c in r]and print()for r in M]and print()),P:=lambda R:None,N:=lambda R:[((p+v)%S[0],(q+w)%S[1],v,w)for p,q,v,w in R],P(R),[R:=N(R)for _ in range(100)],P(R),F:=(sum(1 for p,q,_,_ in R if p<S[0]//2 and q<S[1]//2),sum(1 for p,q,_,_ in R if p>S[0]//2 and q<S[1]//2),sum(1 for p,q,_,_ in R if p<S[0]//2 and q>S[1]//2),sum(1 for p,q,_,_ in R if p>S[0]//2 and q>S[1]//2),),O:=F[0]*F[1]*F[2]*F[3],R:=list(X),D:=False,T:=-1,[(T:=T+1,M:=[[0 for _ in range(S[0])]for _ in range(S[1])],[M[q].__setitem__(p,M[q][p]+1)for p,q,v,w in R],s:="".join(str(x)for m in M for x in m),(D:=True)if"1111111111"in s or S==(11,7)else None,R:=N(R),)for _ in iter(lambda:D,True)],O,T)[-2:])' < example
(12, 0)
```

Part Two only makes sense with real input... Unminified:

```python
print((
# Read from stdin [(px,py,vx,vy)],
X:=open(0).read().splitlines(),
X:=[(a:=x[2:].split(",")) and (b:=a[1].split(" v=")) and (int(a[0]), int(b[0]), int(b[1]), int(a[2])) for x in X],
print(f"{X=}"),

# Get space size depending on input
S:=(11,7) if len(X)==12 else (101,103),
print(f"{S=}"),

R:=list(X),

# Print state
P:=lambda R: (
M:=[[0 for _ in range(S[0])] for _ in range(S[1])],
[M[q].__setitem__(p,M[q][p]+1) for p,q,v,w in R],
[[print(c if c else ".",end="") for c in r] and print() for r in M] and print()),

# Remove for debugging
P:=lambda R: None,

# Next state
N:=lambda R: [((p+v)%S[0],(q+w)%S[1],v,w) for p,q,v,w in R],

# Simulate 100 steps
P(R),
[R:=N(R) for _ in range(100)],
P(R),

# Calculate safetly factor
F:=(
sum(1 for p,q,_,_ in R if p<S[0]//2 and q<S[1]//2), 
sum(1 for p,q,_,_ in R if p>S[0]//2 and q<S[1]//2), 
sum(1 for p,q,_,_ in R if p<S[0]//2 and q>S[1]//2), 
sum(1 for p,q,_,_ in R if p>S[0]//2 and q>S[1]//2), 
),
print(f"{F=}"),

# Part One
O:=F[0]*F[1]*F[2]*F[3],

# Part Two is a bit weird, but I think if we find 10 1's in a row we're done 🎄
# (Your mileage may vary 🤷)
R:=list(X),
D:=False,
T:=-1,
[(
T:=T+1,
M:=[[0 for _ in range(S[0])] for _ in range(S[1])],
[M[q].__setitem__(p,M[q][p]+1) for p,q,v,w in R],
s:="".join(str(x) for m in M for x in m),
(D:=True) if "1111111111" in s or S==(11,7) else None,
R:=N(R),
) for _ in iter(lambda: D, True)],

O, T)[-2:])
```

## 15

```bash
$ python -c 'print((X:=open(0).read().splitlines(),i:=X.index(""),M:=[list(m)for m in X[:i]],I:="".join(X[i:]),p:=lambda M:[[print(x,end="")for x in m]and print()for m in M],p:=lambda M:None,p(M),D:={"^":(-1,0),">":(0,1),"v":(1,0),"<":(0,-1)},init:=lambda M:next((i,j)for i,m in enumerate(M)for j,x in enumerate(m)if x=="@"),M_:=[list(m)for m in M],move:=lambda ci,cj,di,dj:(i:=ci+di,j:=cj+dj,x:=M[i][j],ret:=x!="#",ret:=(ret & move(i,j,di,dj)&(move(i,j+1,di,dj)if x=="["and di!=0 else True)&(move(i,j-1,di,dj)if x=="]"and di!=0 else True))if x in"O[]"else ret,M[i].__setitem__(j,M[ci][cj]),M[ci].__setitem__(cj,"."),ret)[-1],r:=init(M),[(d:=D[ins],B:=[list(m)for m in M],did:=move(*r,*d),did and(r:=(r[0]+d[0],r[1]+d[1]))or(M:=B),)for ins in I],O:=sum(i*100+j for i,m in enumerate(M)for j,x in enumerate(m)if x=="O"),M:=[],[M.append([])or[(M[i].append(x),M[i].append("."))if x=="@"else(M[i].append("["),M[i].append("]"))if x=="O"else(M[i].append(x),M[i].append(x))for j,x in enumerate(m)]for i,m in enumerate(M_)],p(M),r:=init(M),[(d:=D[ins],B:=[list(m)for m in M],did:=move(*r,*d),did and(r:=(r[0]+d[0],r[1]+d[1]))or(M:=B),)for ins in I],T:=sum(i*100+j for i,m in enumerate(M)for j,x in enumerate(m)if x=="["),O,T)[-2:])' < example
(10092, 9021)
```

Today I did it imperative first and then converted. Made me realize how much I appreciate the functional elements the last few days, and converting a solution is pretty boring. Ugly and unminified:

```python
print((
# Read input from stdin
X := open(0).read().splitlines(),
print(X),

# Split map and instructions
i := X.index(""),
M := [list(m) for m in X[:i]],
I := "".join(X[i:]),
print(I),

p := lambda M: [[print(x,end="") for x in m] and print() for m in M],

# Remove for debugging
p := lambda M: None,

p(M),

D := {"^": (-1,0), ">": (0,1), "v": (1,0), "<": (0,-1)},

# Find robot
init := lambda M: next((i,j) for i,m in enumerate(M) for j,x in enumerate(m) if x == "@"),

# Backup
M_ := [list(m) for m in M],

# Step (recursive)
move := lambda ci,cj,di,dj: (
i:=ci+di, j:=cj+dj, x:=M[i][j],
# Return False if wall
ret:=x != "#",
# Try moving the box
ret:=(ret & move(i,j,di,dj) & (move(i,j+1,di,dj) if x=="[" and di!=0 else True) & (move(i,j-1,di,dj) if x=="]" and di!=0 else True)) if x in "O[]" else ret,
M[i].__setitem__(j, M[ci][cj]),
M[ci].__setitem__(cj, "."),
ret)[-1],

# Part One
r := init(M),
[(
d:=D[ins],
B:=[list(m) for m in M],
did:=move(*r,*d),
# Move robot or reset on failure
did and (r:=(r[0]+d[0],r[1]+d[1])) or (M:=B),
) for ins in I],

O := sum(i*100+j for i,m in enumerate(M) for j,x in enumerate(m) if x == "O"),

M:=[],
[M.append([]) or [(M[i].append(x),M[i].append(".")) if x=="@" else (M[i].append("["),M[i].append("]")) if x=="O" else (M[i].append(x),M[i].append(x)) for j,x in enumerate(m)] for i,m in enumerate(M_)],
p(M),

# Part Two
r := init(M),
[(
d:=D[ins],
B:=[list(m) for m in M],
did:=move(*r,*d),
# Move robot or reset on failure
did and (r:=(r[0]+d[0],r[1]+d[1])) or (M:=B),
) for ins in I],

T := sum(i*100+j for i,m in enumerate(M) for j,x in enumerate(m) if x == "["),
O, T)[-2:])
```

## 16

```bash
$ python3 -c 'print((M:=[list(x)for x in open(0).read().splitlines()],s:=next((i,j)for i,m in enumerate(M)for j,x in enumerate(m)if x=="S"),e:=next((i,j)for i,m in enumerate(M)for j,x in enumerate(m)if x=="E"),dd:=__import__("collections").defaultdict,pre:=dd(list),hq:=__import__("heapq"),neighbors:=lambda M,v:[n for n in[None if M[v[0]+v[2]][v[1]+v[3]]=="#"else((v[0]+v[2],v[1]+v[3],v[2],v[3]),1),((v[0],v[1],abs(v[3]),abs(v[2])),1000),((v[0],v[1],-abs(v[3]),-abs(v[2])),1000),]if n is not None],inner:=lambda M,ds,pq:(c:=hq.heappop(pq),[ds.__setitem__(n,d:=c[0]+w)or hq.heappush(pq,(d,n))or pre.__setitem__(n,[c[1]])if c[0]+w<ds[n]else pre[n].append(c[1])if c[0]+w==ds[n]else None for n,w in neighbors(M,c[1])]if c[0]<=ds[c[1]]else None,bool(pq))[-1],dijkstra:=lambda M,s:(ds:=dd(lambda:float("inf")),ds.__setitem__(s,0),pq:=[],hq.heappush(pq,(0,s)),list(iter(lambda:inner(M,ds,pq),False)),ds)[-1],ds:=dijkstra(M,(s[0],s[1],0,1)),O:=min([d for(i,j,_,_),d in ds.items()if(i,j)==e]),best:=[v for v,d in ds.items()if v[0]==e[0]and v[1]==e[1]if d==O],nodes:=set(),dfs:=lambda v:(nodes.add(v),[dfs(p)for p in pre[v]],),[dfs(b)for b in best],nodes:={(i,j)for i,j,_,_ in nodes},T:=len(nodes),O,T)[-2:])' < example
(7036, 45)
```

Unminified:

```python
print((
# Read map from stdin
M := [list(x) for x in open(0).read().splitlines()],
print(f"{M=}"),

# Start/end
s := next((i,j) for i,m in enumerate(M) for j,x in enumerate(m) if x=="S"),
e := next((i,j) for i,m in enumerate(M) for j,x in enumerate(m) if x=="E"),

# Predecessors (for Part Two)
dd := __import__("collections").defaultdict,
pre := dd(list),

# Dijkstra's
hq := __import__("heapq"),

neighbors := lambda M, v: [n for n in [
# Forward
None if M[v[0]+v[2]][v[1]+v[3]]=="#" else ((v[0]+v[2],v[1]+v[3],v[2],v[3]), 1),
# Turns
((v[0],v[1],abs(v[3]),abs(v[2])), 1000),
((v[0],v[1],-abs(v[3]),-abs(v[2])), 1000),
] if n is not None],

inner := lambda M, ds, pq: (
c := hq.heappop(pq),
[ds.__setitem__(n,d:=c[0]+w) or hq.heappush(pq, (d,n)) or pre.__setitem__(n,[c[1]]) if c[0]+w < ds[n] else pre[n].append(c[1]) if c[0]+w == ds[n] else None for n,w in neighbors(M,c[1])] if c[0] <= ds[c[1]] else None,
bool(pq))[-1],

dijkstra := lambda M, s: (

# Distances
ds := dd(lambda: float("inf")),
ds.__setitem__(s,0),

# Priority queue
pq := [],
hq.heappush(pq, (0,s)),

list(iter(lambda: inner(M, ds, pq), False)),

ds)[-1],
ds := dijkstra(M, (s[0],s[1],0,1)),
print(f"{ds=}"),
print(f"{pre=}"),

# Part One
O := min([d for (i,j,_,_),d in ds.items() if (i,j)==e]),

# Part Two, DFS through predecessors
best := [v for v,d in ds.items() if v[0]==e[0] and v[1]==e[1] if d == O],
nodes := set(),
dfs := lambda v: (
nodes.add(v),
[dfs(p) for p in pre[v]],
),
[dfs(b) for b in best],
nodes := {(i,j) for i,j,_,_ in nodes},
T := len(nodes),

O, T)[-2:])
```

## 17

```bash
$ python3 -c 'print((X:=open(0).read().splitlines(),A:=[int(X[0].split(":")[-1])],B:=[int(X[1].split(":")[-1])],C:=[int(X[2].split(":")[-1])],P:=[int(x)for x in X[-1].split(":")[-1].split(",")],ptr:=[0],com:=lambda op:(op if 0<=op<=3 else A[0]if op==4 else B[0]if op==5 else C[0]if op==6 else exit(2)),do:=lambda ins,op,out:((A.__setitem__(0,A[0]//2**com(op)),ptr.__setitem__(0,ptr[0]+2))if ins==0 else(B.__setitem__(0,B[0]^op),ptr.__setitem__(0,ptr[0]+2))if ins==1 else(B.__setitem__(0,com(op)%8),ptr.__setitem__(0,ptr[0]+2))if ins==2 else ptr.__setitem__(0,ptr[0]+2 if A[0]==0 else op)if ins==3 else(B.__setitem__(0,B[0]^C[0]),ptr.__setitem__(0,ptr[0]+2))if ins==4 else(out.append(com(op)%8),ptr.__setitem__(0,ptr[0]+2))if ins==5 else(B.__setitem__(0,A[0]//2**com(op)),ptr.__setitem__(0,ptr[0]+2))if ins==6 else(C.__setitem__(0,A[0]//2**com(op)),ptr.__setitem__(0,ptr[0]+2))if ins==7 else exit(3),),f:=lambda a,b,c:(A.__setitem__(0,a),B.__setitem__(0,b),C.__setitem__(0,c),ptr.__setitem__(0,0),out:=[],list(iter(lambda:(do(P[ptr[0]],P[ptr[0]+1],out),ptr[0]<len(P))[-1],False)),out)[-1],O:=",".join(map(str,f(A[0],B[0],C[0]))),g:=lambda prev,i:(a for a,out in(((prev<<3)+x,f((prev<<3)+x,0,0))for x in range(8))if out==P[-i:]),rec:=lambda gs,i:(xs:=(x for g in gs for x in g),rec([g(x,i+1)for x in xs],i+1)if i<len(P)else next(xs),)[-1],T:=rec([g(0,1)],1),O,T)[-2:])' < example
('5,7,3,0', 117440)
```

Part Two was really hard, but so satisfying. Unminified:

```python
print((
# Read and parse input
X := open(0).read().splitlines(),
print(X),
A := [int(X[0].split(": ")[-1])],
B := [int(X[1].split(": ")[-1])],
C := [int(X[2].split(": ")[-1])],
P := [int(x) for x in X[-1].split(": ")[-1].split(",")],
print(A, B, C, P),

ptr := [0],

# Combo operator
com := lambda op: (
op if 0 <= op <= 3 else
A[0] if op == 4 else
B[0] if op == 5 else
C[0] if op == 6 else
exit(2)
),

# Do work
do := lambda ins, op, out: (
# adv
(A.__setitem__(0,A[0]//2**com(op)), ptr.__setitem__(0,ptr[0]+2)) if ins==0 else
# bxl
(B.__setitem__(0,B[0]^op), ptr.__setitem__(0,ptr[0]+2)) if ins==1 else
# bst
(B.__setitem__(0,com(op)%8), ptr.__setitem__(0,ptr[0]+2)) if ins==2 else
# jnz
ptr.__setitem__(0,ptr[0]+2 if A[0]==0 else op) if ins==3 else
# bxc
(B.__setitem__(0,B[0]^C[0]), ptr.__setitem__(0,ptr[0]+2)) if ins==4 else
# out
(out.append(com(op)%8), ptr.__setitem__(0,ptr[0]+2)) if ins==5 else
# bdv
(B.__setitem__(0,A[0]//2**com(op)), ptr.__setitem__(0,ptr[0]+2)) if ins==6 else
# cdv
(C.__setitem__(0,A[0]//2**com(op)), ptr.__setitem__(0,ptr[0]+2)) if ins==7 else
exit(3),
),

# Collect output
f := lambda a, b, c: (
A.__setitem__(0, a),
B.__setitem__(0, b),
C.__setitem__(0, c),
ptr.__setitem__(0, 0),
out := [],
list(iter(lambda: (do(P[ptr[0]], P[ptr[0]+1], out), ptr[0] < len(P))[-1], False)),
out)[-1],

# Part One
O := ",".join(map(str, f(A[0], B[0], C[0]))),
print("Part One:", O),

# Ok, Part Two was hard...
# We know that the 3 most significant bits should output the last instruction,
# the next 3 should output the next instruction, and so on
# So for each instruction, recursively pick the smallest A that outputs all instructions before and including that instruction

g := lambda prev, i: (a for a,out in (((prev<<3)+x,f((prev<<3)+x,0,0)) for x in range(8)) if out==P[-i:]),

rec := lambda gs,i: (
xs := (x for g in gs for x in g),
rec([g(x,i+1) for x in xs], i+1) if i<len(P) else next(xs),
)[-1],
T := rec([g(0,1)],1),

O, T)[-2:])
```

## 18

```bash
$ python3 -c 'print((X:=[tuple(map(int,x.split(",")))for x in open(0).read().splitlines()],mx:=max(x for x,_ in X),my:=max(y for _,y in X),n_bytes:=12 if(mx,my)==(6,6)else 1024,C:=set(X[:n_bytes]),neighbors:=lambda v:[(x,y)for x,y in[(v[0]+d[0],v[1]+d[1])for d in[(0,1),(1,0),(0,-1),(-1,0)]]if 0<=x<=mx and 0<=y<=my and(x,y)not in C],res:=[],bfs:=lambda s,e:(Q:=__import__("collections").deque([(s,0)]),V:={s},list(iter(lambda:(curr:=Q.popleft(),v:=curr[0],dist:=curr[1],res.append(curr)if v==e else[V.add(n)or Q.append((n,dist+1))for n in neighbors(v)if n not in V],bool(Q)and not bool(res))[-1],False))),bfs((0,0),(mx,my)),O:=res[0][1],g:=((res:=[],C:=set(X[:i+1]),bfs((0,0),(mx,my)),None if res else X[i])[-1]for i in range(len(X))),T:=",".join(map(str,next(x for x in g if x is not None))),O,T)[-2:])' < example
(22, '6,1')
```

Looks like we're doing a graph search again! Unminified:

```python
print((
X := [tuple(map(int,x.split(","))) for x in open(0).read().splitlines()],
print(X),
mx := max(x for x,_ in X),
my := max(y for _,y in X),
print(mx, my),

n_bytes := 12 if (mx,my)==(6,6) else 1024,
C := set(X[:n_bytes]),
print(C),

# BFS to find shortest path
neighbors := lambda v: [(x,y) for x,y in [(v[0]+d[0],v[1]+d[1]) for d in [(0,1),(1,0),(0,-1),(-1,0)]] if 0<=x<=mx and 0<=y<=my and (x,y) not in C],

res := [],
bfs := lambda s, e: (
# [(v, distance to v)]
Q := __import__("collections").deque([(s,0)]),
V := {s},
list(iter(lambda: (
curr := Q.popleft(),
v := curr[0],
dist := curr[1],
res.append(curr) if v == e else [V.add(n) or Q.append((n,dist+1)) for n in neighbors(v) if n not in V],
bool(Q) and not bool(res))[-1], False))
),
bfs((0,0),(mx,my)),
print(res),

# Part One
O := res[0][1],

# Part Two
g := ((
print(i),
res:=[],
C:=set(X[:i+1]),
bfs((0,0),(mx,my)),
None if res else X[i])[-1] for i in range(len(X))),
T := ",".join(map(str,next(x for x in g if x is not None))),

O, T)[-2:])
```

## 19

```bash
$ python3 -c 'print((X:=open(0).read().splitlines(),P:=X[0].split(", "),D:=X[2:],f:=lambda d:(S:=[1]+[0]*len(d),[S.__setitem__(i,S[i]+S[i-len(p)])for i in range(1,len(d)+1)for p in P if len(p)<=i and d[i-len(p):i]==p],S[-1])[-1],R:=[f(d)for d in D],sum(map(bool,R)),sum(R))[-2:])' < example
(6, 16)
```

Today was really fun! Unminified:

```python
print((
X := open(0).read().splitlines(),
P := X[0].split(", "),
D := X[2:],

# Dynamic programming: S[i] = number of ways to build substring d[:i]
f := lambda d: (
S := [1] + [0]*len(d),
[S.__setitem__(i, S[i]+S[i-len(p)])
for i in range(1,len(d)+1)
for p in P
if len(p)<=i and d[i-len(p):i]==p],
S[-1])[-1],

R := [f(d) for d in D],

sum(map(bool,R)),
sum(R))[-2:])
```

## 20

```bash
$ python3 -c 'print((M:=open(0).read().splitlines(),example:=len(M)<100,S:=next((i,j)for i,m in enumerate(M)for j,x in enumerate(m)if x=="S"),E:=next((i,j)for i,m in enumerate(M)for j,x in enumerate(m)if x=="E"),ns:=lambda v:[(v[0]+d[0],v[1]+d[1])for d in[(-1,0),(0,1),(1,0),(0,-1)]if 0<=v[0]+d[0]<len(M)and 0<=v[1]+d[1]<len(M[0])],Q:=__import__("collections").deque([(S,0)]),FS:={S:0},[([(FS.__setitem__(n,v[1]+1),Q.append((n,v[1]+1))if n!=E else None,)for n in ns(v[0])if M[n[0]][n[1]]!="#"and n not in FS])for v in iter(lambda:Q.popleft()if Q else None,None)],Q:=__import__("collections").deque([(E,0)]),FE:={E:0},[([(FE.__setitem__(n,v[1]+1),Q.append((n,v[1]+1))if n!=S else None,)for n in ns(v[0])if M[n[0]][n[1]]!="#"and n not in FE])for v in iter(lambda:Q.popleft()if Q else None,None)],cheat:=lambda t1,ps:(Q:=__import__("collections").deque([(t1,0)]),V:={t1},T:={},[([(V.add(n),T.__setitem__(n,v[1]+1)if M[n[0]][n[1]]!="#"else None,Q.append((n,v[1]+1))if v[1]+1<ps else None,)for n in ns(v[0])if n not in V])for v in iter(lambda:Q.popleft()if Q else None,None)],T)[-1],save:=lambda ps:[((t1,t2),FS[t1],c,FE[t2],cost:=FS[t1]+c+FE[t2],FS[E]-cost)for t1 in FS for t2,c in cheat(t1,ps).items()if t2 in FE],O:=sum(s[-1]>=(50 if example else 100)for s in save(2)),T:=sum(s[-1]>=(50 if example else 100)for s in save(20)),O,T)[-2:])' < example
(1, 285)
```

Gaah, wrestled with this one way too long. Unminified:

```python
print((
M := open(0).read().splitlines(),
print(M),
example := len(M) < 100,

S := next((i,j) for i,m in enumerate(M) for j,x in enumerate(m) if x=="S"),
E := next((i,j) for i,m in enumerate(M) for j,x in enumerate(m) if x=="E"),
print(S, E),

# Strategy, cost of cheat is sum of:
# Shortest path from S to track t1 without cheat
# Shortest path from track t1 to track t2 (allowed to move up to ps with cheat active)
# Shortest path from t2 to E without cheat

# Neighbors
ns := lambda v: [(v[0]+d[0],v[1]+d[1]) for d in [(-1,0),(0,1),(1,0),(0,-1)] if 0<=v[0]+d[0]<len(M) and 0<=v[1]+d[1]<len(M[0])],

# BFS from S to find E (no cheat) and all t1
Q := __import__("collections").deque([(S,0)]),
FS := {S:0},
[(
[(
  FS.__setitem__(n,v[1]+1),
  Q.append((n,v[1]+1)) if n!=E else None,
  ) for n in ns(v[0]) if M[n[0]][n[1]]!="#" and n not in FS]
) for v in iter(lambda: Q.popleft() if Q else None, None)],

print(FS[E]),

# BFS from E to all reachable track cells t2
Q := __import__("collections").deque([(E,0)]),
FE := {E:0},
[(
[(
  FE.__setitem__(n,v[1]+1),
  Q.append((n,v[1]+1)) if n!=S else None,
  ) for n in ns(v[0]) if M[n[0]][n[1]]!="#" and n not in FE]
) for v in iter(lambda: Q.popleft() if Q else None, None)],

print(FE[S]),

# For each t1, find all t2 with cheat active
cheat := lambda t1, ps: (
Q := __import__("collections").deque([(t1,0)]),
V := {t1},
T := {},
[(
[(
  V.add(n),
  T.__setitem__(n,v[1]+1) if M[n[0]][n[1]]!="#" else None,
  Q.append((n,v[1]+1)) if v[1]+1<ps else None,
  ) for n in ns(v[0]) if n not in V]
) for v in iter(lambda: Q.popleft() if Q else None, None)],
T)[-1],

# Calculated ps saved per cheat
save := lambda ps: [((t1,t2), FS[t1], c, FE[t2], cost:=FS[t1]+c+FE[t2], FS[E]-cost) for t1 in FS for t2,c in cheat(t1,ps).items() if t2 in FE],

# Part One
O := sum(s[-1]>=(50 if example else 100) for s in save(2)),

# Part Two
T := sum(s[-1]>=(50 if example else 100) for s in save(20)),

O, T)[-2:])
```

## 21

```bash
$ python3 -c 'print((X:=open(0).read().splitlines(),ds:={(-1,0):"^",(0,1):">",(1,0):"v",(0,-1):"<"},ns:=lambda G,v:[(v[0]+d[0],v[1]+d[1])for d in ds.keys()if 0<=v[0]+d[0]<len(G)and 0<=v[1]+d[1]<len(G[0])],bfs:=lambda G,s:(Q:=__import__("collections").deque([[s]]),res:=__import__("collections").defaultdict(list),[[Q.append(p+[n])or(res[n].append(p+[n])if n not in res or len(p)+1<=len(res[n][0])else None)for n in ns(G,p[-1])if n not in p and G[n[0]][n[1]]!="P"and(sum(abs(bi-ai)and abs(cj-bj)or abs(bj-aj)and abs(ci-bi)for(ai,aj),(bi,bj),(ci,cj)in zip((p+[n]),(p+[n])[1:],(p+[n])[2:]))<2 if len(p)>2 else True)]for p in iter(lambda:Q.popleft()if Q else None,None)],pruned:={},[(left:=k[1]-s[1]<0,v:=bool(p[1][0]-p[0][0]),h:=bool(p[1][1]-p[0][1]),keep:=left and h or not left and v,pruned.__setitem__(k,[p])if keep else None,)for k,P in res.items()for p in P if len(P)>1],res:={**res,**pruned},res)[-1],N:="789\n456\n123\nP0A".splitlines(),Nc:={x:(i,j)for i,r in enumerate(N)for j,x in enumerate(r)if x!="P"},Np:={(i,j):bfs(N,(i,j))for i,r in enumerate(N)for j,x in enumerate(r)if x!="P"},D:="P^A\n<v>".splitlines(),Dc:={x:(i,j)for i,r in enumerate(D)for j,x in enumerate(r)if x!="P"},Dp:={(i,j):bfs(D,(i,j))for i,r in enumerate(D)for j,x in enumerate(r)if x!="P"},solve:=lambda Gc,Gp,s:(seq:=[],T:=__import__("collections").defaultdict(int),[(p:=Gp[Gc[f]][Gc[t]][0]if f!=t else None,ext:=([ds[(bi-ai,bj-aj)]for(ai,aj),(bi,bj)in zip(p,p[1:])]if f!=t else[])+["A"],[T.__setitem__(pair,T[pair]+1)for pair in zip(["A"]+ext,ext)],seq:=seq+ext,)for f,t in zip(s,s[1:])],T)[-1],solveT:=lambda Tp:(T:=__import__("collections").defaultdict(int),[(p:=Dp[Dc[f]][Dc[t]][0]if f!=t else None,ext:=([ds[(bi-ai,bj-aj)]for(ai,aj),(bi,bj)in zip(p,p[1:])]if f!=t else[])+["A"],[T.__setitem__(pair,T[pair]+n)for pair in zip(["A"]+ext,ext)],)for(f,t),n in Tp.items()],T)[-1],do:=lambda s,n:(s:=["A",*s],T:=solve(Nc,Np,s),[T:=solveT(T)for _ in range(n)],sum(T.values()))[-1],O:=sum((c:=do(x,2)*int(x[:-1]),c)[-1]for x in X),T:=sum((c:=do(x,25)*int(x[:-1]),c)[-1]for x in X),O,T)[-2:])' < example
(126384, 154115708116294)
```

Part Two was just evil... 😅 Unminified:

```python
print((
X := open(0).read().splitlines(),
print(X),

# BFS to find all paths between keys in keypads
ds := {(-1,0):"^", (0,1):">", (1,0):"v", (0,-1):"<"},
ns := lambda G, v: [(v[0]+d[0],v[1]+d[1]) for d in ds.keys() if 0<=v[0]+d[0]<len(G) and 0<=v[1]+d[1]<len(G[0])],
bfs := lambda G, s: (
Q := __import__("collections").deque([[s]]),
res := __import__("collections").defaultdict(list),
[[Q.append(p+[n]) or (res[n].append(p+[n]) if n not in res or len(p)+1<=len(res[n][0]) else None) for n in ns(G,p[-1]) if n not in p and G[n[0]][n[1]]!="P"
# Allow only a single 90 degree turn (going zigzag explodes the sequences), intuition: it's good to go straight by spamming A when possible
and (sum(abs(bi-ai) and abs(cj-bj) or abs(bj-aj) and abs(ci-bi) for (ai,aj),(bi,bj),(ci,cj) in zip((p+[n]),(p+[n])[1:],(p+[n])[2:]))<2 if len(p)>2 else True)
]
for p in iter(lambda: Q.popleft() if Q else None, None)],

# I struggled with Part Two, this post helped: https://www.reddit.com/r/adventofcode/comments/1hjgyps/2024_day_21_part_2_i_got_greedyish/
# If we're moving left do horizontal+vertical, if we're moving right do vertical+horizontal
pruned := {},
[(
left:=k[1]-s[1]<0,
v:=bool(p[1][0]-p[0][0]),
h:=bool(p[1][1]-p[0][1]),
keep:=left and h or not left and v,
pruned.__setitem__(k,[p]) if keep else None,
) for k,P in res.items() for p in P if len(P)>1],
res := {**res,**pruned},
res)[-1],

# Numeric keypad
N := """
789
456
123
P0A
""".strip().splitlines(),
print(N),
Nc := {x:(i,j) for i,r in enumerate(N) for j,x in enumerate(r) if x!="P"},
Np := {(i,j):bfs(N,(i,j)) for i,r in enumerate(N) for j,x in enumerate(r) if x!="P"},

# Directional keypad
D := """
P^A
<v>
""".strip().splitlines(),
print(D),
Dc := {x:(i,j) for i,r in enumerate(D) for j,x in enumerate(r) if x!="P"},
Dp := {(i,j):bfs(D,(i,j)) for i,r in enumerate(D) for j,x in enumerate(r) if x!="P"},

# Which buttons should be pressed on dpad to get s?
solve := lambda Gc, Gp, s: (
seq := [],
T := __import__("collections").defaultdict(int),
[(
p:=Gp[Gc[f]][Gc[t]][0] if f!=t else None,
ext:=([ds[(bi-ai,bj-aj)] for (ai,aj),(bi,bj) in zip(p,p[1:])] if f!=t else [])+["A"],
[T.__setitem__(pair,T[pair]+1)for pair in zip(["A"]+ext, ext)],
seq:=seq+ext,
) for f,t in zip(s,s[1:])],
T)[-1],

# Only count transition types so we don't run out of memory
solveT := lambda Tp: (
T := __import__("collections").defaultdict(int),
[(
p:=Dp[Dc[f]][Dc[t]][0] if f!=t else None,
ext:=([ds[(bi-ai,bj-aj)] for (ai,aj),(bi,bj) in zip(p,p[1:])] if f!=t else [])+["A"],
[T.__setitem__(pair,T[pair]+n)for pair in zip(["A"]+ext, ext)],
) for (f,t),n in Tp.items()],
T)[-1],

# Solve for each robot, so 123A is transformed into dpad seq s1, then s is transformed to s2 and so on
do := lambda s, n: (
s := ["A", *s],
T := solve(Nc, Np, s),
[T := solveT(T) for _ in range(n)],
sum(T.values()))[-1],

# Part One
print("Part One..."),
O := sum((
print("Solving:",x),
c:=do(x,2)*int(x[:-1]),
print("Complexity:",c),
c)[-1] for x in X),
print("Answer:",O,"\n"),

# Part Two
print("Part Two..."),
T := sum((
print("Solving:",x),
c:=do(x,25)*int(x[:-1]),
print("Complexity:",c),
c)[-1] for x in X),
print("Answer:",T,"\n"),

O, T)[-2:])
```
