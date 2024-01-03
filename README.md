JDK Profiling Tester
=========================

A tool to test the stability of JFR and AsyncGetCallTrace by using them
to profile a set of benchmarks with small capturing intervals.
It tests AsyncGetCallTrace using AsyncProfiler.

The "aim" is to find sporadic JVM crashes that result in `hs_err` files.

Setup
-----

```sh
git clone https://github.com/parttimenerd/jdk-profiling-tester.git --recursive
```

Usage
-----

This script has no dependencies other than python3 (version >= 3.8).

```sh
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

For example test async-profiler 2.9:

```sh
./main.py profile asgct-2.9 openjdk-server-fastdebug
```

Or test JFR:

```sh
./main.py profile jfr openjdk-server-fastdebug
```

Or test the current async-profiler master:

```sh
./main.py profile asgct-2.9 openjdk-server-fastdebug
```

The JDK and profiler are built if needed. Use the `./main.py build`
to rebuild specific jdks.

Supported JDKS
--------------

- `openjdk`: OpenJDK master
- `openjdk21u`: JDK 21
- your own JDK, just pass the path to the `images` or `jdk` folder

Benchmarks
----------

- [dacapo](https://github.com/dacapobench/dacapobench)
- [renaissance](https://renaissance.dev/)

Profile Command
---------------

```sh
usage: main.py profile [-h] [--benchmarks BENCHMARKS] [--interval INTERVAL] [--timeout TIMEOUT]
                       [--program_timeout PROGRAM_TIMEOUT] [--parallelism PARALLELISM]
                       [--iterations ITERATIONS]
                       profiler_pattern jdk_pattern

positional arguments:
  profiler_pattern      profilers: jfr, asgct
  jdk_pattern           glob style patterns for jdks, supports commas

options:
  -h, --help            show this help message and exit
  --benchmarks BENCHMARKS
                        glob style patterns for benchmarks, supports commas
  --interval INTERVAL, -i INTERVAL
                        Interval for obtaining call traces, might be an interval, e.g. '0.1ms,0.2ms'
  --timeout TIMEOUT, -t TIMEOUT
                        Timeout for benchmarked programs
  --program_timeout PROGRAM_TIMEOUT
                        Timeout that the benchmarked program is passed if supported
  --parallelism PARALLELISM, -p PARALLELISM
                        > 1 for parallel profiling, parallel for every profiler
  --iterations ITERATIONS, -c ITERATIONS
                        number of iterations per profiler
```

License
-------
MIT, Copyright 2023 SAP SE or an SAP affiliate company, Johannes Bechberger
and trace-validation contributors

*This project is a prototype of the [SapMachine](https://sapmachine.io) team
at [SAP SE](https://sap.com)*