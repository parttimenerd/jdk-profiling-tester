#! python3
import argparse
import fnmatch
import multiprocessing
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Iterable, Any, Callable

BASE_PATH = Path(__file__).parent
JDKS_PATH = BASE_PATH / "jdks"
BENCH_PATH = BASE_PATH / "benchmarks"


def pad_left(s: str, count: int) -> str:
    return " " * (count - len(s)) + s


def pad_right(s: str, count: int) -> str:
    return s + " " * (count - len(s))


def match(pattern: str, value: str) -> bool:
    return any(fnmatch.fnmatch(value, pat) for pat in pattern.split(","))


@dataclass(frozen=True)
class Benchmark:
    name: str
    file: Union[Path, str]
    """ either jar file or class name """
    classpath: Optional[Path] = None
    arguments: Optional[List[str]] = None
    cleanup_files: List[str] = None

    @staticmethod
    def for_class(classpath: Path, klass: str) -> "Benchmark":
        assert classpath.exists()
        return Benchmark(klass, klass, classpath)

    @staticmethod
    def for_jar(jar: Path, arguments: List[str], name: str = None, cleanup_files: List[str] = None) -> "Benchmark":
        assert jar.exists()
        return Benchmark(name or f"{' '.join(jar.name.split('.')[:-1])} {' '.join(arguments)}", jar,
                         arguments=arguments, cleanup_files=cleanup_files)

    def java_arguments(self, run_seconds: int = 100000000) -> List[str]:
        args: List[str] = []
        if self.classpath:
            args.extend(["-cp", str(self.classpath)])
        if isinstance(self.file, Path):
            args.extend(["-jar", str(self.file)])
        else:
            args.append(self.file)
        args.extend(arg.replace("$RUN_SECONDS", str(run_seconds)) for arg in self.arguments)
        return args

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def cleanup(self, folder: Path):
        for f in self.cleanup_files or []:
            if (p := folder / f).exists():
                shutil.rmtree(p)


def download(url, target_filename):
    os.makedirs(BENCH_PATH, exist_ok=True)
    target_path = BENCH_PATH / target_filename
    if not target_path.exists():
        print(f"Downloading {target_filename} from {url}")
        urllib.request.urlretrieve(url, BENCH_PATH / target_filename)


download("https://github.com/renaissance-benchmarks/renaissance/releases/download/v0.14.0/renaissance-gpl-0.14.0.jar",
         "renaissance.jar")
download("https://downloads.sourceforge.net/project/dacapobench/9.12-bach-MR1/dacapo-9.12-MR1-bach.jar",
         "dacapo.jar")

BENCHMARKS: List[Benchmark] = [
                                  Benchmark.for_jar(BENCH_PATH / "dacapo.jar", [x], cleanup_files=["scratch"])
                                  for x in
                                  ["avrora", "fop", "h2", "jython", "lusearch", "lusearch-fix", "pmd", "sunflow",
                                   "tomcat", "xalan"]
                              ] + [
                                  Benchmark.for_jar(BENCH_PATH / "renaissance.jar",
                                                    [x, "--run-seconds", "$RUN_SECONDS"], f"renaissance {x}")
                                  for x in
                                  ['scrabble', 'page-rank', 'future-genetic', 'akka-uct', 'movie-lens', 'scala-doku',
                                   'chi-square',
                                   'fj-kmeans', 'rx-scrabble', 'finagle-http', 'reactors', 'dec-tree',
                                   'scala-stm-bench7',
                                   'naive-bayes', 'als', 'par-mnemonics', 'scala-kmeans',
                                   'philosophers', 'log-regression', 'gauss-mix', 'mnemonics', 'dotty',
                                   'finagle-chirper']
                              ]


def get_benchmarks(pattern: str) -> List[Benchmark]:
    """ supports glob style patterns, e.g. "dacapo*" matches all dacapo benchmarks """
    return [bench for bench in BENCHMARKS if match(pattern, bench.name)]


class JDKDebugLevel(Enum):
    RELEASE = "release"
    FASTDEBUG = "fastdebug"


class JDKVariant(Enum):
    SERVER = "server"
    CLIENT = "client"
    ZERO = "zero"


@dataclass(frozen=True)
class JDKBuildType:
    variant: JDKVariant
    debug_level: JDKDebugLevel

    def build_params(self) -> List[str]:
        """ parameters for configure """
        params: List[str] = []
        if shutil.which("ccache") is not None:
            params.append("--enable-ccache")
        if platform.system() == "Darwin":
            params.extend(["--disable-precompiled-headers", "--disable-warnings-as-errors"])
        params.extend(["--with-debug-level=" + self.debug_level.value, "--with-jvm-variants=" + self.variant.value])
        return params

    def build_folder_ending(self) -> str:
        return f"{self.variant.value}-{self.debug_level.value}"

    def __str__(self) -> str:
        return self.build_folder_ending()

    @staticmethod
    def parse(name: str) -> "JDKBuildType":
        variant, dbg = name.split("-")
        return JDKBuildType(JDKVariant[variant.upper()], JDKDebugLevel[dbg.upper()])


@dataclass(frozen=True)
class JDK:
    name: str
    base_path: Path

    def build_folder(self, bt: JDKBuildType) -> Path:
        return self._build_folder(bt)

    def has_build(self, bt: JDKBuildType) -> bool:
        return len(list(self.base_path.glob(f"build/*{bt}"))) > 0

    def _configure(self, bt: JDKBuildType):
        subprocess.check_call(["bash", "configure", *bt.build_params()], cwd=self.base_path, stdout=subprocess.DEVNULL)

    def _build_folder(self, bt: JDKBuildType) -> Path:
        return Path(list(self.base_path.glob(f"build/*{bt}"))[0])

    def _make(self, bt: JDKBuildType):
        subprocess.check_call(["make", f"CONF={self._build_folder(bt).name}", "images"], cwd=self.base_path,
                              stdout=subprocess.DEVNULL)

    def build(self, bt: JDKBuildType):
        self._configure(bt)
        self._make(bt)

    def get(self, bt: JDKBuildType) -> "JDKBuild":
        if not self.has_build(bt):
            self.build(bt)
        return IncludedJDKBuild(self, bt)

    def __str__(self):
        return self.name


class JDKBuild:

    def bin_path(self) -> Path:
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def supports_asgct2(self) -> bool:
        raise NotImplementedError()

    def java_path(self) -> Path:
        return self.bin_path() / "java"


@dataclass(frozen=True)
class IncludedJDKBuild(JDKBuild):
    jdk: JDK
    type: JDKBuildType

    def __str__(self) -> str:
        return f"{self.jdk}-{self.type}"

    def bin_path(self) -> Path:
        return self.jdk.build_folder(self.type) / "images" / "jdk" / "bin"

    def supports_asgct2(self) -> bool:
        return "asgct2" in self.jdk.name


@dataclass(frozen=True)
class CustomJDKBuild(JDKBuild):
    path: Path
    """ path to jdk folder """

    def __str__(self) -> str:
        return str(self.path)

    def bin_path(self) -> Path:
        return self.path / "bin"

    def supports_asgct2(self) -> bool:
        return True


# available JDKS
JDKS = {p.name: JDK(p.name, p) for p in JDKS_PATH.iterdir() if p.is_dir()}


def get_jdk(name: str) -> JDK:
    return JDKS[name]


def parse_jdk_build_name(name: str) -> Tuple[JDK, JDKBuildType]:
    jdk, bt = name.split("-", maxsplit=1)
    return get_jdk(jdk), JDKBuildType.parse(bt)


def available_jdk_build_names(pattern: str) -> List[str]:
    return [name for jdk in JDKS for variant in JDKVariant for dbg in JDKDebugLevel
            if match(pattern, (name := f"{jdk}-{variant.value}-{dbg.value}"))]


def build(jdk_build_name: str):
    jdk, bt = parse_jdk_build_name(jdk_build_name)
    jdk.build(bt)


def get_jdk_build(name: str) -> JDKBuild:
    if "/" in name:
        path = Path(name)
        if (p := (path / "images" / "jdk")).exists():
            path = p
        elif (p := (path / "jdk")).exists():
            path = p
        else:
            assert path.name == "jdk"
        return CustomJDKBuild(path.absolute())
    jdk, bt = parse_jdk_build_name(name)
    return jdk.get(bt)


def get_jdk_builds(pattern: str) -> List[JDKBuild]:
    pats = pattern.split(",")
    ret: List[JDKBuild] = []
    for pat in pats:
        if "/" in pat:
            ret.append(get_jdk_build(pat))
        else:
            ret.extend(get_jdk_build(name) for name in available_jdk_build_names(pat))
    return ret


# use micro seconds, finer units do not make any sense
def usecs(time_val: str) -> int:
    split = next(re.finditer("[a-z]", time_val)).start()
    val, unit = time_val[:split], time_val[split:]
    value = float(val)
    if unit == "us":
        return int(value)
    elif unit == "ms":
        return int(value * 1000)
    elif unit == "s":
        return int(value * 1000 * 1000)
    elif unit == "m":
        return int(value * 1000 * 1000 * 60)
    elif unit == "h":
        return int(value * 1000 * 1000 * 60 * 60)
    else:
        assert False, f"{unit} not supported"


def parse_interval(time_val: str) -> int:
    if "," in time_val:
        start, end = map(usecs, time_val.split(","))
        return random.randrange(start, end)
    return usecs(time_val)


@dataclass
class ProfilerConfig:
    _interval: str
    """ passed in by the user """
    timeout: float
    program_timeout: float
    parallelism: int = 1

    def interval(self) -> float:
        """ returns the interval in usecs """
        return parse_interval(self._interval)


class ResultBase:

    def error_rate(self) -> float:
        return self.errors() * 1.0 / len(self) if len(self) else 0

    def errors(self) -> int:
        return sum(r.errors() for r in self._results())

    def __len__(self) -> int:
        return self._len(self._results())

    def _results(self, d=None) -> Iterable["ResultBase"]:
        d = d or self.results
        if isinstance(self.results, list):
            return self.results
        if isinstance(self.results, dict):
            return self._results(list(d.values()))
        assert False

    @staticmethod
    def _errors(sub: Iterable["ResultBase"]) -> int:
        return sum(s.errors() for s in sub)

    @staticmethod
    def _len(sub: Iterable["ResultBase"]) -> int:
        return sum(len(s) for s in sub)


@dataclass
class Result(ResultBase):
    hs_err: Optional[Path]

    def errors(self) -> int:
        return 1 if self.hs_err else 0

    def __len__(self) -> int:
        return 1


@dataclass
class IteratedResult(ResultBase):
    results: List[Result]


@dataclass
class Results:
    config: ProfilerConfig
    jdks: List[JDKBuild]
    benchmarks: List[Benchmark]
    results: Dict[JDKBuild, Dict[Benchmark, Result]]


@dataclass
class IteratedResults:
    profiler: "Profiler"
    config: ProfilerConfig
    jdks: List[JDKBuild]
    benchmarks: List[Benchmark]
    results: Dict[JDKBuild, Dict[Benchmark, IteratedResult]] = None
    start: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if self.results is None:
            self.results = {jdk: {benchmark: IteratedResult([]) for benchmark in self.benchmarks} for jdk in self.jdks}

    def add(self, jdk: JDKBuild, benchmark: Benchmark, result: Result):
        self.results[jdk][benchmark].results.append(result)

    def combined_jdk_results(self) -> List[IteratedResult]:
        return [IteratedResult(list(x for benchmark in self.benchmarks for x in self.results[jdk][benchmark].results))
                for jdk in self.jdks]

    def __str__(self) -> str:
        """
        Printed table looks like:

                  openjdk-debug-release    ...
        All                10  / 0.1
        Bench 1            ...
        """

        def format_iresult(iresult: IteratedResult) -> str:
            return f"{len(iresult)} * {iresult.error_rate():.3f}"

        lines: List[str] = []
        max_benchmark_width = max(len(benchmark.name) for benchmark in self.benchmarks)

        def add_line(header: str, parts: List[Any]):
            lines.append(", ".join([pad_right(header, max_benchmark_width),
                                    *(pad_left(str(p), len(str(jdk))) for jdk, p in zip(self.jdks, parts))]))

        add_line("", self.jdks)
        add_line("All", [format_iresult(ji) for ji in self.combined_jdk_results()])
        for benchmark in self.benchmarks:
            add_line(benchmark.name, [format_iresult(self.results[jdk][benchmark]) for jdk in self.jdks])

        return f"Profiler {self.profiler.name} tests for {int(time.time() - self.start)}s\n" \
               f"with config {self.config}\n\n" + "\n".join(lines)

    def write(self, path: Path):
        path.write_text(str(self) + "\n")


def Profiler_run(self: "Profiler", config: ProfilerConfig, jdk: JDKBuild, benchmark: Benchmark,
                 result_base_folder: Path):
    return jdk, benchmark, self.run(config, jdk, benchmark, result_base_folder)


def Profiler_run2(args):
    return Profiler_run(*args)


class Profiler:

    def __init__(self, name: str):
        self.name = name

    def run_all_iterated(self, config: ProfilerConfig, jdks: List[JDKBuild], benchmarks: List[Benchmark],
                         result_base_folder: Path, max_iterations: int):
        os.makedirs(result_base_folder, exist_ok=True)
        results_file = result_base_folder / "results.csv"
        results: IteratedResults = IteratedResults(self, config, jdks, benchmarks)
        try:
            def handler(jdk, benchmark, result):
                results.add(jdk, benchmark, result)
                if result.hs_err:
                    print(results)
                    results.write(results_file)

            self.run_all(config, jdks, benchmarks, result_base_folder, max_iterations, handler)

        finally:
            print()
            print(results)
            print("\n")
            results.write(results_file)

    def run_all(self, config: ProfilerConfig, jdks: List[JDKBuild], benchmarks: List[Benchmark],
                result_base_folder: Path, max_count: int,
                handler: Callable[[JDKBuild, Benchmark, Result], None]):
        if config.parallelism == 1:
            for i in range(max_count):
                for jdk in jdks:
                    for benchmark in benchmarks:
                        handler(jdk, benchmark, self.run(config, jdk, benchmark, result_base_folder))
        else:
            with multiprocessing.Pool(processes=config.parallelism) as pool:
                for jdk, benchmark, result in pool.imap_unordered(Profiler_run2,
                                                                  [(self, config, jdk, benchmark, result_base_folder)
                                                                   for i in range(max_count)
                                                                   for jdk in jdks
                                                                   for benchmark in benchmarks],
                                                                  chunksize=1):
                    handler(jdk, benchmark, result)

    def run(self, config: ProfilerConfig, jdk: JDKBuild, benchmark: Benchmark,
            result_base_folder: Path) -> Result:
        base_folder = result_base_folder / str(jdk) / benchmark.name
        os.makedirs(base_folder, exist_ok=True)
        count = len(list(base_folder.iterdir())) if config.parallelism == 1 else time.time_ns()
        folder = base_folder / str(count)
        os.makedirs(folder, exist_ok=True)
        res = self._run(config, jdk.java_path(),
                        benchmark.java_arguments(int(config.program_timeout)), folder, benchmark)
        return res

    def _run(self, config: ProfilerConfig,
             java_binary: Path, bench_args: List[str], folder: Path, benchmark: Benchmark) -> Result:
        env = self._env(java_binary.parent)
        cmd = [java_binary, *self._java_arguments(config), f"-XX:ErrorFile=hs_err.log",
               "-XX:+UnlockDiagnosticVMOptions", "-XX:+DebugNonSafepoints", *bench_args]
        try:
            print(" ".join(str(c).replace(str(BASE_PATH), ".") for c in cmd))
            try:
                out = subprocess.check_output(cmd, env=env, cwd=folder, stderr=subprocess.PIPE,
                                              timeout=config.timeout).decode()
            except subprocess.TimeoutExpired as tex:
                print(tex)
            finally:
                self.cleanup_and_check(folder)
            return Result(None)
        except subprocess.CalledProcessError as x:
            # we currently only look for hs_err files
            # if "Digest validation failed" in x.stdout.decode()
            # or "java.lang.reflect.InvocationTargetException" in x.stderr.decode():
            #    return Result(None)
            hs_err = folder / "hs_err.log"
            if not hs_err.exists():
                return Result(None)
            (folder / "cmd.sh").write_text(f"""
# with timeout of {config.interval()}us run 
{" ".join(map(str, cmd))}
""")
            print(f"hs_err file: {hs_err}")
            print("------")
            print("".join(hs_err.open().readlines()[:20]))
            print("Error: " + "".join(x.stderr.decode().split()[:20]))
            return Result(hs_err)
        finally:
            benchmark.cleanup(folder)

    def _java_arguments(self, config: ProfilerConfig) -> List[str]:
        raise NotImplementedError()

    def uses_asgct2(self) -> bool:
        return False

    def compatible(self, jdk: JDKBuild) -> bool:
        return jdk.supports_asgct2() or not self.uses_asgct2()

    @staticmethod
    def _env(jdk_folder: Path) -> Dict[str, str]:
        env = os.environ.copy()
        env["JAVA_HOME"] = str(jdk_folder)
        env["PATH"] = f"{jdk_folder}/bin:{env['PATH']}"
        return env

    def build(self):
        pass

    def cleanup_and_check(self, folder: Path):
        pass


class AsyncProfiler(Profiler):

    def __init__(self, name: str, base_path: Path, asgct2: bool):
        super(AsyncProfiler, self).__init__(name)
        self.asgct2 = asgct2
        self.base_path = base_path

    def _java_arguments(self, config: ProfilerConfig) -> List[str]:
        return [
            f"-agentpath:{self.base_path}/build/libasyncProfiler.so=start,flat=10000,interval={config.interval()}us,"
            f"traces=1,event=cpu"]

    def uses_asgct2(self) -> bool:
        return self.asgct2

    def build(self):
        subprocess.check_call(["make"], cwd=self.base_path, stdout=subprocess.DEVNULL)


class JFR(Profiler):

    def __init__(self):
        super(JFR, self).__init__("jfr")

    def _java_arguments(self, config: ProfilerConfig) -> List[str]:
        return [f"-XX:StartFlightRecording=filename=flight.jfr,"
                f"jdk.ExecutionSample#period={max(1, round(config.interval() / 1000))}ms"]

    def uses_asgct2(self) -> bool:
        return False

    def cleanup_and_check(self, folder: Path):
        out = subprocess.check_output(["jfr", "print", "--events", "jdk.ExecutionSample", "flight.jfr"], cwd=folder)
        assert "jdk.ExecutionSample" in out.strip().decode()
        os.remove(folder / "flight.jfr")


PROFILERS = [JFR(),
             AsyncProfiler("asgct2", BASE_PATH / "profilers" / "asgct2-async-profiler", asgct2=True),
             AsyncProfiler("asgct", BASE_PATH / "profilers" / "async-profiler", asgct2=False)]


def get_profilers(pattern: str) -> List[Profiler]:
    return [profiler for profiler in PROFILERS if match(pattern, profiler.name)]


def profile(config: ProfilerConfig, profilers: List[Profiler],
            jdks: List[JDKBuild], benchmarks: List[Benchmark],
            result_base_folder: Path, max_iterations: int):
    try:
        if not len(profilers):
            warnings.warn("no profilers given")
            return
        if not len(jdks):
            warnings.warn("no jdks given")
            return
        for profiler in profilers:
            profiler.build()
            comp_jdks = [jdk for jdk in jdks if profiler.compatible(jdk)]
            if not len(comp_jdks):
                warnings.warn(f"no jdks given that are compatible to the profiler {profiler}")
            profiler.run_all_iterated(config,
                                      comp_jdks,
                                      benchmarks, result_base_folder / profiler.name,
                                      max_iterations)
    except KeyboardInterrupt as tex:
        print("aborted", file=sys.stderr)


def cli():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(title="subcommands", help="additional help")

    def handle_build(args):
        print("Build jdks")
        for n in args.jdk:
            build(n)

    build_parser = subparsers.add_parser("build", help="build jdks")
    build_parser.add_argument("jdk", metavar="jdk-variant-dbg", nargs="*",
                              help="see available_jdks for possible arguments")
    build_parser.set_defaults(func=handle_build)

    def handle_av_jdks(args):
        for x in available_jdk_build_names(args.pattern):
            print(x)

    av_jdks_parser = subparsers.add_parser("jdks", help="available JDK builds")
    av_jdks_parser.add_argument("pattern", help="glob style pattern (supports commas), "
                                                "e.g. 'openjdk*' matches all openjdk builds",
                                nargs="?")
    av_jdks_parser.set_defaults(pattern="*", func=handle_av_jdks)

    def handle_av_benchmarks(args):
        for x in get_benchmarks(args.pattern):
            print(x.name)

    av_benchmarks_parser = subparsers.add_parser("benchmarks", help="available benchmarks")
    av_benchmarks_parser.add_argument("pattern", help="glob style pattern (supports commas), "
                                                      "e.g. 'dacapo*' matches all dacapo benchmarks",
                                      nargs="?")
    av_benchmarks_parser.set_defaults(pattern="*", func=handle_av_benchmarks)

    def handle_profile(args):
        profile(ProfilerConfig(args.interval, args.timeout, args.program_timeout, args.parallelism),
                get_profilers(args.profiler_pattern),
                get_jdk_builds(args.jdk_pattern),
                get_benchmarks(args.benchmarks),
                BASE_PATH / "results" / str(int(time.time())),
                args.iterations)

    profile_parser = subparsers.add_parser("profile", help="profile benchmarks with jdks and profilers")
    profile_parser.add_argument("profiler_pattern",
                                help=f"profilers: {', '.join(profiler.name for profiler in PROFILERS)}")
    profile_parser.add_argument("jdk_pattern", help="glob style patterns for jdks, supports commas")
    profile_parser.add_argument("--benchmarks", help="glob style patterns for benchmarks, supports commas")
    profile_parser.add_argument("--interval", "-i", help="Interval for obtaining call traces, might be an interval, "
                                                         "e.g. '0.1ms,0.2ms'")
    profile_parser.add_argument("--timeout", "-t", help="Timeout for benchmarked programs", type=float)
    profile_parser.add_argument("--program_timeout", help="Timeout that the benchmarked program is "
                                                          "passed if supported", type=float)
    profile_parser.add_argument("--parallelism", "-p", help="> 1 for parallel profiling, parallel for every profiler",
                                type=int)
    profile_parser.add_argument("--iterations", "-c", help="number of iterations per profiler", type=int)
    profile_parser.set_defaults(profiler_pattern="*", jdk_pattern="*", benchmarks="*",
                                interval="100us", timeout=200, program_timeout=30, parallelism=1,
                                iterations=10, func=handle_profile)

    args = parser.parse_args(sys.argv[1:] if sys.argv[0].endswith(".py") else sys.argv)
    args.func(args)


if __name__ == '__main__':
    cli()
