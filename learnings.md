# Solving AoC with one Python expression

This year, I solved all puzzles using a single Python expression: https://github.com/oskaerik/aocg24 (Unminified versions are included from day 8 and forward)

I started doing day 1 in Go, but thought "this is a oneliner in Python!", and here we are...

## What's an expression?

If you can do an `eval(<expression>)`, it's an expression. That is, you can't use semicolons to have multiple statements. And no loops, try/excepts, assignment/import statements, etc.

So... what can we do?

Well, we can `print()` stuff... Just kidding, we're programmers, right? We can do whatever we want!

## Control flow aka tuples, tuples everywhere!

So you want to print two things? Well:

```python
(print("hello"), print("world"))
```

Nice, now we're doing two things in one expression! This gives us a nice outline for our solutions:

```python
print((
<do stuff>,
p1, p2)[-2:])
```

This will print a tuple `(p1, p2)`. Now we just need to replace the `<do stuff>` with some boilerplate so `p1` and `p2` contains the answers to the puzzle.

Combine this with some inline `... if ... else ...` and you have your control flow figured out.

You can also do control flow with `and/or` to spice it up a little:

```python
lst and print(lst) or print("empty")
```

## Do you even loop?

Some puzzles require loops. But loops are not expressions. So we can either 1) not loop, or 2) be smart. And the smart thing is using **comprehensions**!

This basically replaces a for-loop:

```python
[print(i) for i in range(10)]
```

Or crazy stuff like a double for loop with filtering:

```python
{(i, j):i * j for i in range(10) for j in range(1, i) if i % j == 0}
```

## But what about while loops?

I did BFS more times than I can count this year. And while BFSing you typically do a while loop, right?

Fret not, yet again we can be clever. `iter(callable, sentinel)` to the rescue!

You pass it a callable and it will keep calling the callable until it sees the sentinel value, then stop:

```python
iter(lambda x=[1, 2, 3]: x.pop() if x else None, None)
```

If you squint a little, you now have something like this:

```python
def f():
    x = [1, 2, 3]
    while x:
        yield x.pop()
```

## Variables?

Ah, we can't do assignment statements. But we can walrus!

```python
(a := 1, b := 2, print(a + b))
```

Or alternatively:

```python
locals().__setitem__("a", 1)
```

Or even `globals()` if we're really brave.

## Sure, but how can I solve the puzzles without importing anything?

Yeah, you have to implement the entire stdlib yourself unfortunately.

Haha, got you again!

```python
__import__("collections").defaultdict(int)
```

## Putting it all together

All right, let's outline a BFS:

```python
print((

bfs := lambda start: (
    queue := __import__("collections").deque([start]),
    visited := {start},
    [[(visited.add(n), queue.append(n)) for n in neighbors(v) if n not in visited] for v in iter(lambda: queue.popleft() if queue else None, None)],
),

...,

res)[-1])
```

So, yeah. That's basically how to solve AoC in one expression. Oh yeah, and the input can be read from stdin with:

```python
open(0).read().splitlines()
```
