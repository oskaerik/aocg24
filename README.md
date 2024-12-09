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
