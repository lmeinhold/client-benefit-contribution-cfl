import abc
import dataclasses
import pathlib
import uuid
from datetime import datetime

import jsonpickle


def _new_run_id() -> str:
    return str(uuid.uuid4())


class Logger:
    def __init__(self, algorithm, dataset: str, model: str, rounds: int, epochs: int, adapter=None):
        self.algorithm = algorithm
        self.dataset = dataset
        self.model = model
        self.rounds = rounds
        self.epochs = epochs
        self.adapter = adapter if adapter else ConsoleAdapter()
        self.run_id = _new_run_id()

    def log_run_data(self):
        entry = LogMetadata(
            run_id=self.run_id,
            algorithm=self.algorithm,
            dataset=self.dataset,
            model=self.model,
            rounds=self.rounds,
            epochs=self.epochs,
        )
        self.adapter.write_metadata(entry)

    def log_server_metrics(self, round: int, stage="train", **kwargs):
        entry = LogEntry(
            timestamp=datetime.now(),
            log_type="server",
            stage=stage,
            client_id=None,
            round=round,
            epoch=None,
            metrics=kwargs
        )
        self.adapter.write_log_entry(entry)

    def log_client_metrics(self, client_id: str, round: int, epoch: int, stage="train", **kwargs):
        entry = LogEntry(
            timestamp=datetime.now(),
            log_type="client",
            stage=stage,
            client_id=client_id,
            round=round,
            epoch=epoch,
            metrics=kwargs
        )
        self.adapter.write_log_entry(entry)

    def attach(self, adapter: "LoggerAdapter"):
        self.adapter = adapter


@dataclasses.dataclass
class LogEntry:
    timestamp: datetime
    log_type: str
    stage: str
    client_id: None | str
    round: int
    epoch: int
    metrics: dict[str, any]


@dataclasses.dataclass
class LogMetadata:
    run_id: str
    algorithm: str
    dataset: str
    model: str
    rounds: int
    epochs: int


class LoggerAdapter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write_log_entry(self, log_entry: "LogEntry"):
        raise NotImplemented()

    @abc.abstractmethod
    def write_metadata(self, meta_entry: "LogMetadata"):
        raise NotImplemented()


class ConsoleAdapter(LoggerAdapter):
    def write_metadata(self, meta_entry: "LogMetadata"):
        print(f"""Run: {meta_entry.run_id}""")

    def write_log_entry(self, log_entry: "LogEntry"):
        loc = log_entry.client_id if log_entry.log_type == "client" else "server"
        metrics = ", ".join([f"{k}={v}" for k, v in log_entry.metrics.items()])
        print(f"[{log_entry.timestamp}  r {log_entry.round}, e {log_entry.epoch} | {metrics}")


class JsonAdapter(LoggerAdapter):
    def __init__(self, outdir: pathlib.Path, run_id: str):
        if not outdir.parent.exists():
            outdir.parent.mkdir(parents=True)
        elif outdir.exists() and not outdir.is_dir():
            raise FileNotFoundError(f"Output {outdir} is not a directory")

        self.outfile_metrics = outdir.joinpath(run_id + "_metrics.json").open("w")
        self.outfile_meta = outdir.joinpath(run_id + "_meta.json").open("w")

    def write_log_entry(self, log_entry: "LogEntry"):
        line = jsonpickle.encode(log_entry, unpicklable=False) + "\n"
        self.outfile_metrics.write(line)
        self.outfile_metrics.flush()

    def write_metadata(self, meta_entry: "LogMetadata"):
        content = jsonpickle.encode(meta_entry, unpicklable=False)
        self.outfile_meta.write(content)
        self.outfile_meta.flush()
        self.outfile_meta.close()


class CompositeAdapter(LoggerAdapter):
    """Wrapper class for using multiple LoggerAdapters"""
    def __init__(self, *adapters):
        self.adapters = adapters

    def write_log_entry(self, log_entry: "LogEntry"):
        for adapter in self.adapters:
            adapter.write_log_entry(log_entry)

    def write_metadata(self, meta_entry: "LogMetadata"):
        for adapter in self.adapters:
            adapter.write_metadata(meta_entry)
