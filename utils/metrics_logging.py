import abc
import uuid
from datetime import datetime


def _new_run_id() -> str:
    return str(uuid.uuid4())


class Logger:
    def __init__(self, algorithm, dataset: str, model: str, adapter=None):
        self.algorithm = algorithm
        self.dataset = dataset
        self.model = model
        self.adapter = adapter if adapter else ConsoleAdapter()
        self.run_id = _new_run_id()

    def log_server_metrics(self, round: int, stage="train", **kwargs):
        entry = LogEntry(
            run_id=self.run_id,
            timestamp=datetime.now(),
            algorithm=self.algorithm,
            dataset=self.dataset,
            model=self.model,
            log_type="server",
            stage=stage,
            client_id=None,
            round=round,
            epoch=None,
            metrics=kwargs
        )

    def log_client_metrics(self, client_id: str, round: int, epoch: int, stage="train", **kwargs):
        entry = LogEntry(
            run_id=self.run_id,
            timestamp=datetime.now(),
            algorithm=self.algorithm,
            dataset=self.dataset,
            model=self.model,
            log_type="client",
            stage=stage,
            client_id=client_id,
            round=round,
            epoch=epoch,
            metrics=kwargs
        )


class LogEntry:
    def __init__(self, run_id: str, timestamp: datetime, algorithm: str, dataset: str, model: str, log_type: str,
                 stage: str, client_id: None | str, round: int, epoch: None | int, metrics: dict[str, any]):
        self.run_id = run_id
        self.timestamp = timestamp
        self.algorithm = algorithm
        self.dataset = dataset
        self.model = model
        self.log_type = log_type
        self.stage = stage
        self.client_id = client_id
        self.round = round
        self.epoch = epoch
        self.metrics = metrics


class LoggerAdapter(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def write_log_entry(self, log_entry: "LogEntry"):
        raise NotImplemented()


class ConsoleAdapter(LoggerAdapter):
    def write_log_entry(self, log_entry: "LogEntry"):
        loc = log_entry.client_id if log_entry.log_type == "client" else "server"
        metrics = ", ".join([f"{k}={v}" for k, v in log_entry.metrics.items()])
        print(f"[{log_entry.timestamp} {log_entry.run_id} {loc}] {log_entry.algorithm} {log_entry.model} on\
         {log_entry.dataset} | r {log_entry.round}, e {log_entry.epoch} | {metrics}")
