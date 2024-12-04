# Advent of Code (Golf) 2024

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
