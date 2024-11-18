Start the program with
```
    python main.py <ptx_file>
```

-----
# Input
Need to pass in time if kernel are fired multiple times (without mem copy)
# Observation
If there is a loop in the kernel, need to input the possible iteration of it before running.
 - Downside: cannot handle multiple loop. Assume only one loop in the kernel.
 - Hard to detech coalescing and loop unrolling.

----
To get cuda ptx file.
```
    cuda -ptx <cu_file> -o <output>
```