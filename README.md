# Advent of Code (Golf) 2024

Rules:

1. One single-line Python expression
2. Print a tuple `(answer1, answer2)`

## 01

```bash
$ python3 -c 'print((l:=tuple(map(sorted,zip(*(map(int,l.strip().split())for l in __import__("sys").stdin)))),sum(abs(a-b)for a,b in zip(*l)),sum(a*sum(1 for x in l[1] if x==a)for a in l[0]))[1:])' < example
(11, 31)
```

## 02

```bash
$ python3 -c 'print(((l:=[[int(x)for x in l.strip().split()]for l in __import__("sys").stdin]),tuple(sum([any([all([1<=b-a<=3 for a,b in list(zip(l,l[1:]))])for l in r])for r in[[l[:i]+l[i+d:]for i,_ in enumerate(l)]for l in[list(reversed(x))if x[0]>x[-1] else x for x in l]]])for d in(0,1)))[1])' < example
(2, 4)
```

## 03

```bash
$ python3 -c 'print(((x:="do()"+"".join(__import__("sys").stdin),f:=lambda d:sum(int(x[0])*int(x[1])for x in[x.split(",")for x in"".join((x.split("do()",maxsplit=1)+[""])[1]for x in x.split(d)).split("mul(")for x in x.split(")")]if len(x)==2 and all(x.isnumeric()for x in x))),f("🐍"),f("don'\''t()"))[-2:])' < example
(161, 48)
```

## 04

```bash
$ python3 -c 'print((x:=[x.strip()for x in __import__("sys").stdin],sum(x=="XMAS"for x in["".join(x[r+a*i][c+b*i]for i in range(4))for r,_ in enumerate(x)for c,_ in enumerate(x[0])for a,b in [(a,b)for a in(-1,0,1)for b in(-1,0,1)if a or b]if 0<=r+a*3<len(x)and 0<=c+b*3<len(x[0])]),len([x for x in [sum(x)for x in [["".join((x[r-a][c-b],x[r][c],x[r+a][c+b]))=="MAS"for a,b in[(a,b)for a in(-1,1)for b in(-1,1)]if 0<=r-a<len(x)and 0<=r+a<len(x)and 0<=c-b<len(x[0])and 0<=c+b<len(x[0])]for r,_ in enumerate(x)for c,_ in enumerate(x[0])]]if x==2]))[1:])' < example
(18, 9)
```

## 05

```bash
$ python3 -c 'print(globals().__setitem__("a",lambda k,v:globals().__setitem__(k,v))or a("l",[l.strip()for l in __import__("sys").stdin])or a("r",[r.split("|")for r in l[:l.index("")]])or a("p",[u.split(",")for u in l[l.index("")+1:]])or a("d",lambda g,u:u in V or V.add(u)or([d(g,v)for v in g.get(u,[])]or 1)and O.append(u))or a("t",lambda g:a("V",set())or a("O",[])or ([d(g,n)for n in g]or 1)and O[::-1])or a("G",{k:[v for u,v in r if u==k]for k,_ in r})or a("c",lambda G,O:a("P",{n:i for i,n in enumerate(O)})or all(u not in P or v not in P or P[u]<P[v]for u in G for v in G[u]))or(sum(int(u[len(u)//2])for u in p if c(G,u)),sum(int(t({k:[x for x in v if x in u]for k,v in G.items()if k in u})[len(u)//2])for u in p if not c(G,u))))' < example
(143, 123)
```
