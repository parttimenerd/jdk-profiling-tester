#! /usr/bin/python3
import os.path
import subprocess
from dataclasses import dataclass
from typing import Callable, List


@dataclass
class Result:
    duration: float
    file_size: float


def mean(arr: List[float]) -> float:
    return sum(arr) / len(arr)


def run_benchmark(arguments: str) -> Result:
    durations = []
    file_sizes = []
    for i in range(5):
        env = os.environ.copy()
        env["LANG"] = "en.US"
        out = subprocess.check_output(
            f"perf stat -- java {arguments} -jar benchmarks/dacapo.jar avrora fop h2 jython lusearch pmd -t 8 -n 3 > /dev/null",
            shell=True, env=env, stderr=subprocess.STDOUT)
        time = float(out.decode().split(" seconds time elapsed")[0].split("\n")[-1].strip())
        durations.append(time)
        file_sizes.append(os.path.getsize("file.jfr") if (os.path.exists("file.jfr")) else 0)
    return Result(mean(durations), mean(file_sizes))


def run_async_profiler(interval: float) -> Result:
    inter = (str(round(interval * 1000.0)) + 'm') if (interval >= 0.001) else (str(round(interval * 1000000.0)) + 'u')
    return run_benchmark(
        f"-agentpath:./profilers/async-profiler/build/libasyncProfiler.so=start,interval={inter}s,event=cpu,file=file.jfr,jfr -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints")


def run_jfr(interval: float, profile: str) -> Result:
    return run_benchmark(
        f"-XX:StartFlightRecording=filename=file.jfr,jdk.ExecutionSample#period={max(1, round(interval * 1000))}ms,jdk.NativeExecutionSample#period={max(1, round(interval * 1000))}ms,settings={profile} -XX:+UnlockDiagnosticVMOptions -XX:+DebugNonSafepoints")


def create_table(callable: Callable[[float], Result], intervals: List[float]) -> str:
    base_line = run_benchmark("")
    table = [["interval", "time", "overhead", "file_size"], [0, base_line.duration, 0, 0]]
    for interval in intervals:
        try:
            result = callable(interval)
            table.append(
                [interval, result.duration, (result.duration - base_line.duration) / base_line.duration,
                 result.file_size])
            print("\n".join(",".join(map(str, row)) for row in table))
        except BaseException as ex:
            print(ex)
    return "\n".join(",".join(map(str, row)) for row in table)


def create_tables(intervals: List[float]) -> str:
    return "\n\n".join([
        "jfr cpu_and_mem_profiling.jfc\n" + create_table(
            lambda interval: run_jfr(interval, "cpu_and_mem_profiling.jfc"), intervals),
        "jfr cpu_profiling.jfc\n" + create_table(lambda interval: run_jfr(interval, "cpu_profiling.jfc"), intervals),
        "jfr profile.jfc\n" + create_table(lambda interval: run_jfr(interval, "profile.jfc"), intervals),
        "jfr default.jfc\n" + create_table(lambda interval: run_jfr(interval, "default.jfc"), intervals),
        "async-profiler\n" + create_table(run_async_profiler, intervals)])


print(create_tables(
    [0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011,
     0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002,
     0.001, 0.0005, 0.0001]))
