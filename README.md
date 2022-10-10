JDK Profiling Tester
=========================

A tool to test the stability of JFR and AsyncGetStackTrace/AsyncGetCallTrace by using them
to profile a set of benchmarks with small capturing intervals.
It tests AsyncGetCallTrace and the proposed AsyncGetCallTrace2 
([JEP draft](https://openjdk.org/jeps/8284289),
[demo](https://github.com/parttimenerd/asgst-demo/)) using AsyncProfiler.

The "aim" is to find sporadic JVM crashes that result in `hs_err` files.

Usage
-----

This script has no dependencies other than python3 (version >= 3.8).

```
> ./main.py
usage: main.py [-h] {build,jdks,benchmarks,profile} ...

optional arguments:
  -h, --help            show this help message and exit

subcommands:
  {build,jdks,benchmarks,profile}
                        additional help
    build               build jdks
    jdks                available JDK builds
    benchmarks          available benchmarks
    profile             profile benchmarks with jdks and profilers
```

For example test AsyncGetCallTrace with the asgst prototype version via

```sh
./main.py profile asgct asgst-server-fastdebug
```

Supported JDKS
--------------

- `openjdk`: OpenJDK master
- `openjdk18u`: JDK 18
- `asgst`: detached OpenJDK master with AsyncGetStackTrace support
- your own JDK, just pass the path to the `images` or `jdk` folder

Benchmarks
----------

- [dacapo](https://github.com/dacapobench/dacapobench)
- [renaissance](https://renaissance.dev/)


License
-------
MIT