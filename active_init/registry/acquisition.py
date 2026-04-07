import torch
from torch import Tensor
from inspect import signature

from copy import deepcopy
from omegaconf import OmegaConf
import gpytorch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from botorch.models.model import Model
from botorch.utils.sampling import draw_sobol_samples
from botorch.acquisition.utils import get_optimal_samples
from botorch.acquisition import AcquisitionFunction
from botorch.acquisition.monte_carlo import qPosteriorStandardDeviation

from botorch.models.utils.gpytorch_modules import MIN_INFERRED_NOISE_LEVEL
from active_init.acquisition.active_learning import (
    qNegIntegratedPosteriorVariance,
    qExpectedPredictiveInformationGain, 
)

from active_init.acquisition.bayesian_active_learning import (
    qHyperparameterInformedPredictiveExploration,
    qBayesianActiveLearningByDisagreement,
)

ACQUISITION_REGISTRY = {
    "nipv": qNegIntegratedPosteriorVariance,
    "qPSTD": qPosteriorStandardDeviation,
    "epig": qExpectedPredictiveInformationGain,
    "hipe": qHyperparameterInformedPredictiveExploration,
    "bald": qBayesianActiveLearningByDisagreement,
}

MC_SAMPLING_STRATS = {
    "sobol": draw_sobol_samples,
}


def get_acquisition_function(
    objective: callable,
    acq_name: str,
    model: Model,
    acq_kwargs: dict,
    bounds: Tensor,
    X_pending: Tensor | None = None,
    mc_strategy: str | None = None,
) -> AcquisitionFunction:
    acq_class = ACQUISITION_REGISTRY[acq_name]
    if "map_model" not in signature(acq_class).parameters and "map_model" in acq_kwargs.keys():
        del acq_kwargs["map_model"]
    
    if "bounds" in signature(acq_class).parameters:
        acq_kwargs["bounds"] = bounds
    
    if "num_mc_points" in acq_kwargs:
        num_mc_points = acq_kwargs.pop("num_mc_points")
        mc_strategy = acq_kwargs.pop("mc_strategy", "sobol")
        if mc_strategy == "opt":
            mc_points, _ = get_optimal_samples(
                model,
                bounds=bounds,
                num_optima=num_mc_points,
                raw_samples=512,
                num_restarts=1
            )
            mc_points = mc_points.unsqueeze(-2)
        elif mc_strategy == "sobol":
            mc_points = draw_sobol_samples(
                n=num_mc_points,
                q=1,
                bounds=bounds,
            )
            #mc_points = torch.cat((mc_points * 0.35 + 0.46, mc_points * 0.2 + 0.1))
            if not model._is_fully_bayesian:
                mc_points = mc_points.squeeze(-2)

        else:
            raise ValueError("MC strategy not implemented.")
        
        acq_kwargs["mc_points"] = mc_points
    
    acqf = acq_class(model=model, X_pending=X_pending, **acq_kwargs)
    return acqf
