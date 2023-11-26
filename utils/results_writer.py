import pandas as pd
import torch


class ResultsWriter:
    def __init__(self):
        self.metrics = {}

    def write(self, **kwargs) -> "ResultsWriter":
        for k, v in kwargs.items():
            self._write_single_metric(k, v)
        return self

    def _write_single_metric(self, name: str, value):
        if name in self.metrics:
            self.metrics[name].append(value)
        else:
            try:
                if isinstance(value, torch.Tensor) and value.get_device() >= 0:
                    print(f"WARNING! value {value} for key {name} is on gpu!")
            except:
                pass

            self.metrics[name] = [value]

    def as_dataframe(self):
        return pd.DataFrame(self.metrics)

    def save(self, path):
        df = self.as_dataframe()
        df.to_csv(path, index=False)
