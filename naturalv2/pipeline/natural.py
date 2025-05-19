import pandas as pd
from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Any, Dict, List


class PipelineStage(ABC):
    """Base class for stages in a pipeline."""
    
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.experiment = None
        self.source_name = None
        self.estimator_type = None
        self.outcome = None
        self.save_path = None
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the input data and return transformed data."""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the stage's processing."""
        pass


class NATURALPipeline:
    
    def __init__(
            self, 
            experiment,
            source_name,
            estimator_type,
            outcome,
            save_path,
        ):
        self.experiment = experiment
        self.source_name = source_name
        self.estimator_type = estimator_type
        self.outcome = outcome
        self.save_path = save_path
        self.stages: List[PipelineStage] = []
        self.data_flow: Dict[str, int] = {}
        
    def add_stage(self, stage: PipelineStage) -> 'NATURALPipeline':
        """Add a processing stage to the pipeline."""
        stage.experiment = self.experiment
        stage.source_name = self.source_name
        stage.estimator_type = self.estimator_type
        stage.outcome = self.outcome
        stage.save_path = self.save_path
        self.stages.append(stage)
        return self
        
    def run(self, initial_data: pd.DataFrame) -> pd.DataFrame:
        """Run the pipeline on the input data."""
        current_data = initial_data
        
        for stage in self.stages:
            current_data = stage.process(current_data)
            self.data_flow.update(stage.get_stats())
            
        return current_data