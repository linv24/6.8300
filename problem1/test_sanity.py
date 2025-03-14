from __future__ import annotations
import pydantic
import torch
import jaxtyping
import random
import torch.nn as nn
import pytest
from safetensors.torch import load_file, save_file
from pydantic_yaml import parse_yaml_file_as
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    # Not sure why, but I (Adriano) got some issues when I was using `pytest .`
    # To make testing super easy like that, we just try multiple import paths... one of them
    # should eventually work. Feel free to modify this as needed or `cd <this dir> && pytest .`.
    from problem_1_siren import SineLayer, SIREN
    from problem_1_gradients import gradient, divergence, laplace
    from problem_1_mlp import MLP
    from utils import set_seed
except ImportError:
    try:
        import problem1.problem_1_siren as SineLayer
        import problem1.problem_1_siren as SIREN
        import problem1.problem_1_gradients as gradient
        import problem1.problem_1_gradients as divergence
        import problem1.problem_1_gradients as laplace
        import problem1.problem_1_mlp as MLP
        from utils import set_seed
    except ImportError:
        raise ImportError("Failed to import the necessary modules. Please ensure that the files are in the correct directory.")

"""
[You can ignore this file if you want]

This file is used to test the sanity of the code. It does not guarantee that your
implementation is correct nor does it guarantee that your implementation is
incorrect, but if you are failing these tests you probably will not get full credit,
while if you are passing you are probably on the right track.

These use random seeds (and law of large numbers) to ensure that tests are as
deterministic as possible, but it's possible that this is not 100%. Read through
the specifics of your failing test if you are failing any test.

Grading is done by looking at your code for conceptual correctness and verifying
that the models you train are sufficiently accurate (that the images look as good or
better than the examples in `should_look_like/` and PSNR is as good or better than
the examples in `should_look_like/`).
"""

# NOTE: do NOT modify `REGRESS_ON_CURRENT_RESULTS` to `True` unless you want to get
# the WRONG tensor values in `test_data/` and thereby the wrong test results.
REGRESS_ON_CURRENT_RESULTS = False

def list2dict(l: list[torch.Tensor], prefix: str = "") -> dict[str, torch.Tensor]:
    return {f"{prefix}{i}": output for i, output in enumerate(l)}
def dict2list(d: dict[str, torch.Tensor], prefix: str = "") -> list[torch.Tensor]:
    return [d[f"{prefix}{i}"] for i in range(len(d))]

class HelperTestForward:
    """
    Helper class to just test a forward pass for a model (generically)
    since all models need to work with forward passes. It expected the form
    of nn.Module that we are using in `problem1_siren.py`, `problem1_gradients.py`,
    These take in coordinates and return the outputs and a gradient-enabled clone
    of the coordinates.
    """

    regress_on_current_results: bool = REGRESS_ON_CURRENT_RESULTS
    prefix: Optional[str] = None
    outputs_coords: bool = False

    INPUT_KEY = "input"
    OUTPUT_KEY = "output"
    COORDS_KEY = "coords"

    def tensors_filename(self) -> Path:
        return Path(__file__).parent / "test_data" / f"{self.prefix}.safetensors" # fmt: skip

    def load_io(
        self, inputs: List[torch.Tensor], models: List[torch.nn.Module]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        if self.regress_on_current_results:
            outputs = [model(input) for model, input in zip(models, inputs)]
            if self.outputs_coords:
                outputs = [output[0] for output in outputs] # ignore the coord
            self.save_io_to_file(outputs, inputs)
            return list(zip(outputs, inputs))
        else:
            assert self.tensors_filename().exists(), f"File {self.tensors_filename()} does not exist" # fmt: skip
            return self.load_io_from_file()

    def load_io_from_file(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        data = load_file(self.tensors_filename())
        assert isinstance(data, dict)
        num_samples = len(data)
        assert num_samples % 2 == 0
        num_samples //= 2
        outputs = [data[f"output_{i}"] for i in range(num_samples)]
        inputs = [data[f"input_{i}"] for i in range(num_samples)]
        return list(zip(outputs, inputs))

    def save_io_to_file(self, outputs: List[torch.Tensor], inputs: List[torch.Tensor]): # fmt: skip
        assert len(outputs) == len(inputs)
        obj = {}
        for i in range(len(outputs)):
            obj[f"output_{i}"] = outputs[i]
            obj[f"input_{i}"] = inputs[i]
        save_file(obj, self.tensors_filename()) # fmt: skip

    def setup_io(
        self,
        outputs_inputs_expected: List[
            Tuple[
                jaxtyping.Float[torch.Tensor, "N output_dim"],
                jaxtyping.Float[torch.Tensor, "N input_dim"],
            ]
            | Tuple[Path, Path]
        ],
        models: List[torch.nn.Module],
    ):
        self.outputs_inputs_expected: List[
            Tuple[
                jaxtyping.Float[torch.Tensor, "N output_dim"],
                jaxtyping.Float[torch.Tensor, "N input_dim"],
            ]
        ] = []
        for output, input in outputs_inputs_expected:
            if isinstance(output, Path):
                output = load_file(output)[self.OUTPUT_KEY]
            if isinstance(input, Path):
                input = load_file(input)[self.INPUT_KEY]
            self.outputs_inputs_expected.append((output, input))
        self.models = models
        assert len(self.outputs_inputs_expected) == len(self.models)
    

    @classmethod
    def generate_realistic_inputs(cls, seed: int, models: List[torch.nn.Module]) -> List[torch.Tensor]:
        set_seed(seed)
        inputs = [torch.rand(10, model.in_features) * 2 - 1 for model in models]
        inputs.extend([torch.zeros(5, model.in_features) for model in models])
        inputs.extend([torch.ones(10, model.in_features) for model in models])
        inputs.extend([-torch.ones(10, model.in_features) for model in models])
        return inputs

    def test_forward_no_throw(self):
        for model, (output, input) in zip(self.models, self.outputs_inputs_expected):
            with torch.no_grad():
                model(input)  # throws

    def test_forward_shape(self):
        for model, (expected_output, input) in zip(self.models, self.outputs_inputs_expected):
            with torch.no_grad():
                if self.outputs_coords:
                    output, coords = model(input)
                    assert output.shape == expected_output.shape, f"(1) output.shape={output.shape}, expected_output.shape={expected_output.shape}" # fmt: skip
                    assert coords.shape == input.shape, f"(2) coords.shape={coords.shape}, input.shape={input.shape}" # fmt: skip
                else:
                    output = model(input)
                    assert output.shape == expected_output.shape, f"(3) output.shape={output.shape}, expected_output.shape={expected_output.shape}" # fmt: skip

    def test_forward_value(self):
        for model, (expected_output, input) in zip(self.models, self.outputs_inputs_expected):
            with torch.no_grad():
                if self.outputs_coords:
                    output, coords = model(input)
                    coords_expected_mxae = torch.abs(coords - input).max().item()
                    output_expected_mxae = torch.abs(output - expected_output).max().item()
                    assert torch.allclose(coords, input), f"(1) mxae={coords_expected_mxae}" # fmt: skip
                    assert torch.allclose(output, expected_output), f"(2) mxae={output_expected_mxae}" # fmt: skip
                else:
                    output = model(input)
                    mxae = torch.abs(output - expected_output).max().item()
                    assert torch.allclose(output, expected_output), f"(3) mxae={mxae}" # fmt: skip


################################ CODE TESTERS ################################
class TestSineLayer(HelperTestForward):
    regress_on_current_results: bool = REGRESS_ON_CURRENT_RESULTS
    prefix: str = "sine_layer"
    outputs_coords: bool = False

    @pytest.fixture(autouse=True)
    def setup(self):
        _models = [
            SineLayer(
                in_features=3, out_features=32, bias=True, is_first_layer=True, omega_0=30
            ),
            SineLayer(
                in_features=4, out_features=1, bias=False, is_first_layer=True, omega_0=2
            ),
            SineLayer(
                in_features=1, out_features=5, bias=True, is_first_layer=True, omega_0=30
            ),
        ]
        # This should be consistent due to the seed
        set_seed(0)
        _models[0].linear.weight.data = torch.rand(*tuple(_models[0].linear.weight.data.shape))
        _models[1].linear.weight.data = torch.rand(*tuple(_models[1].linear.weight.data.shape))
        _models[2].linear.weight.data = torch.rand(*tuple(_models[2].linear.weight.data.shape))
        _models[0].linear.weight.bias = torch.zeros(*tuple(_models[0].linear.bias.shape))
        assert _models[1].linear.bias is None
        _models[2].linear.weight.bias = -torch.ones(*tuple(_models[2].linear.bias.shape))
        models: List[nn.Module] = [
            _models[0],
            _models[0],
            _models[0],
            # ...
            _models[1],
            _models[1],
            _models[1],
            # ...
            _models[2],
            _models[2],
            _models[2],
        ]
        set_seed(0)
        inputs: List[torch.Tensor] = [
            torch.rand(10, 3),
            torch.zeros(1, 3),
            torch.ones(1000, 3),
            # ...
            torch.rand(10, 4),
            torch.zeros(1, 4),
            -torch.ones(1000, 4),
            # ...
            torch.rand(10, 1),
            torch.zeros(1, 1),
            -torch.ones(1000, 1),
        ]
        outputs_inputs = self.load_io(inputs, models)
        assert len(outputs_inputs) == len(models)
        assert all(isinstance(output, torch.Tensor) for output, _ in outputs_inputs) # fmt: skip
        assert all(isinstance(input, torch.Tensor) for _, input in outputs_inputs) # fmt: skip
        super().setup_io(outputs_inputs, models)
        assert len(outputs_inputs) == len(models)
        assert all(isinstance(output, torch.Tensor) for output, _ in self.outputs_inputs_expected) # fmt: skip
        assert all(isinstance(input, torch.Tensor) for _, input in self.outputs_inputs_expected) # fmt: skip

    # Pytest doesn't find some of these :/
    def test_forward_no_throw(self):
        super().test_forward_no_throw()

    def test_forward_shape(self):
        super().test_forward_shape()

    def test_forward_value(self):
        super().test_forward_value()

    def test_initialization(self):
        """
        Do a probabilistic test that your initialization scheme is close to
        the proper distribution. This should use the law of large numbers to
        generally test whether your scheme is correct irrespective of the
        actual detail of how you did it. It will work both for deterministic
        and non-deterministic initializations.
        """
        omega_0 = 30
        io_feats = [(8, 8), (5, 1), (1, 5), (1, 1), (11, 12), (10, 10)]
        is_firsts = [True, True, True, False, False, False]

        # Heuristic based on CLT
        num_tries = 2**16
        atol_expansion_factor = 3 # Allow for some amount more atol just to be safe
        allowable_atol = atol_expansion_factor * torch.sqrt(torch.tensor(1/num_tries)).item() # around 1e-2

        # Regress with correct results
        if self.regress_on_current_results:
            expected_means = []
            expected_variances = []
            for i, ((in_feat, out_feat), is_first) in enumerate(list(zip(io_feats, is_firsts))):
                sls = [SineLayer(in_features=in_feat, out_features=out_feat, bias=True, is_first_layer=is_first, omega_0=omega_0) for _ in range(num_tries)] # fmt: skip
                layer_weights = torch.stack([s.linear.weight.data for s in sls])
                mean_layer_weights = torch.mean(layer_weights, dim=0)
                var_layer_weights = torch.var(layer_weights, dim=0)
                expected_means.append(mean_layer_weights)
                expected_variances.append(var_layer_weights)
            save_file(list2dict(expected_variances, "expected_variances"), "test_data/expected_variances.safetensors")
            save_file(list2dict(expected_means, "expected_means"), "test_data/expected_means.safetensors")

        # Load correct results
        _expected_variances = load_file("test_data/expected_variances.safetensors") # fmt: skip
        _expected_means = load_file("test_data/expected_means.safetensors") # fmt: skip
        expected_variances = dict2list(_expected_variances, "expected_variances") # fmt: skip
        expected_means = dict2list(_expected_means, "expected_means") # fmt: skip

        # Compare with correct results
        for i, ((in_feat, out_feat), is_first) in enumerate(list(zip(io_feats, is_firsts))):
            sls = [SineLayer(in_features=in_feat, out_features=out_feat, bias=True, is_first_layer=is_first, omega_0=omega_0) for _ in range(num_tries)] # fmt: skip
            layer_weights = torch.stack([s.linear.weight.data for s in sls])
            assert layer_weights.ndim == 3, f"layer_weights.ndim={layer_weights.ndim}" # fmt: skip
            assert layer_weights.shape == (num_tries, out_feat, in_feat), f"layer_weights.shape={layer_weights.shape}" # fmt: skip
            mean_layer_weights = torch.mean(layer_weights, dim=0)
            var_layer_weights = torch.var(layer_weights, dim=0)
            assert mean_layer_weights.ndim == 2, f"mean_layer_weights.ndim={mean_layer_weights.ndim}" # fmt: skip
            assert var_layer_weights.ndim == 2, f"var_layer_weights.ndim={var_layer_weights.ndim}" # fmt: skip
            assert expected_variances[i].shape == var_layer_weights.shape, f"expected_variances[i].shape={expected_variances[i].shape}, var_layer_weights.shape={var_layer_weights.shape}" # fmt: skip
            assert expected_means[i].shape == mean_layer_weights.shape, f"expected_means[i].shape={expected_means[i].shape}, mean_layer_weights.shape={mean_layer_weights.shape}" # fmt: skip
            mae_var = torch.abs(expected_variances[i] - var_layer_weights).max().item() # fmt: skip
            mae_mean = torch.abs(expected_means[i] - mean_layer_weights).max().item() # fmt: skip
            assert torch.allclose(expected_means[i], mean_layer_weights, atol=allowable_atol), f"max_abs_error_mean={mae_mean:9e}" # fmt: skip
            assert torch.allclose(expected_variances[i], var_layer_weights, atol=allowable_atol), f"max_abs_error_var={mae_var:9e}" # fmt: skip


class TestMLP(HelperTestForward):
    regress_on_current_results: bool = REGRESS_ON_CURRENT_RESULTS
    prefix: str = "mlp"
    outputs_coords: bool = True

    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(0)
        _models = []
        for _ in range(100):
            _models.append(MLP(
                in_features=torch.randint(1, 3, (1,)).item(),
                out_features=torch.randint(1, 3, (1,)).item(),
                hidden_features=torch.randint(1, 32, (1,)).item(),
                hidden_layers=torch.randint(1, 10, (1,)).item(),
                bias=bool(torch.randint(0, 2, (1,)).item()),
                activation=random.choice(["ReLU", "Tanh", "GELU"])
            ))
        # Re-initialize all parameters for each model
        set_seed(0)
        for model in _models:
            for param in model.parameters():
                nn.init.normal_(param, mean=0.0, std=0.1) # eh - seed
        # Ensure we have at least one of each activation type
        activations = set(model.activation for model in _models)
        assert len(activations) == 3, "Need at least one of each activation type (ReLU, Tanh, GELU)" # fmt: skip
        assert "ReLU" in activations, "Need at least one ReLU activation" # fmt: skip
        assert "Tanh" in activations, "Need at least one Tanh activation" # fmt: skip
        assert "GELU" in activations, "Need at least one GELU activation" # fmt: skip
        
        # NOTE: this is OK because of seeding
        # Ensure we have at least one with in_features=1 and one with in_features>1
        in_features_sizes = set(model.in_features for model in _models)
        assert 1 in in_features_sizes, "Need at least one model with in_features=1" # fmt: skip
        assert any(size > 1 for size in in_features_sizes), "Need at least one model with in_features>1" # fmt: skip
        
        # Ensure we have at least one with out_features=1 and one with out_features>1
        out_features_sizes = set(model.out_features for model in _models)
        assert 1 in out_features_sizes, "Need at least one model with out_features=1" # fmt: skip
        assert any(size > 1 for size in out_features_sizes), "Need at least one model with out_features>1" # fmt: skip
        
        # Ensure we have at least one with bias and one without bias
        bias_values = set(model.bias for model in _models)
        assert True in bias_values, "Need at least one model with bias=True" # fmt: skip
        assert False in bias_values, "Need at least one model with bias=False" # fmt: skip
        
        # Ensure we have at least one with hidden_layers=1
        hidden_layers_values = set(model.hidden_layers for model in _models)
        assert 1 in hidden_layers_values, "Need at least one model with hidden_layers=1" # fmt: skip
        
        # Stride is the entire batch
        set_seed(0)
        inputs = self.generate_realistic_inputs(0, _models)
        models = _models * 4 # everyone gets the 1st, then the 2nd, ... (so everyone gets 4)
        
        outputs_inputs_expected = self.load_io(inputs, models)
        super().setup_io(outputs_inputs_expected, models)

    # Pytest doesn't find some of these :/
    def test_forward_no_throw(self):
        super().test_forward_no_throw()

    def test_forward_shape(self):
        super().test_forward_shape()

    def test_forward_value(self):
        super().test_forward_value()


class TestSIREN(HelperTestForward):
    regress_on_current_results: bool = REGRESS_ON_CURRENT_RESULTS
    prefix: str = "siren"
    outputs_coords: bool = True

    @pytest.fixture(autouse=True)
    def setup(self):
        set_seed(0)
        _models = []
        for _ in range(100):
            _models.append(SIREN(
                in_features=torch.randint(1, 3, (1,)).item(),
                out_features=torch.randint(1, 3, (1,)).item(),
                hidden_features=torch.randint(1, 32, (1,)).item(),
                hidden_layers=torch.randint(1, 10, (1,)).item(),
                bias=bool(torch.randint(0, 2, (1,)).item()),
                last_layer_linear=bool(torch.randint(0, 2, (1,)).item()),
                first_omega_0=(20**torch.randn(1).item()),
                hidden_omega_0=(20**torch.randn(1).item()),
            ))
        # Re-initialize all parameters for each model
        set_seed(0)
        for model in _models:
            for param in model.parameters():
                nn.init.normal_(param, mean=0.0, std=0.1) # eh - seed (don't use init, tested seperately)

        # Asserts from before
        in_features_sizes = set(model.in_features for model in _models)
        assert 1 in in_features_sizes, "Need at least one model with in_features=1" # fmt: skip
        assert any(size > 1 for size in in_features_sizes), "Need at least one model with in_features>1" # fmt: skip
        
        # Ensure we have at least one with out_features=1 and one with out_features>1
        out_features_sizes = set(model.out_features for model in _models)
        assert 1 in out_features_sizes, "Need at least one model with out_features=1" # fmt: skip
        assert any(size > 1 for size in out_features_sizes), "Need at least one model with out_features>1" # fmt: skip
        
        # Ensure we have at least one with bias and one without bias
        bias_values = set(model.bias for model in _models)
        assert True in bias_values, "Need at least one model with bias=True" # fmt: skip
        assert False in bias_values, "Need at least one model with bias=False" # fmt: skip
        
        # Ensure we have at least one with hidden_layers=1
        hidden_layers_values = set(model.hidden_layers for model in _models)
        assert 1 in hidden_layers_values, "Need at least one model with hidden_layers=1" # fmt: skip
        
        # Stride is the entire batch
        set_seed(0)
        inputs = self.generate_realistic_inputs(0, _models)
        models = _models * 4 # everyone gets the 1st, then the 2nd, ... (so everyone gets 4)
        
        outputs_inputs_expected = self.load_io(inputs, models)
        super().setup_io(outputs_inputs_expected, models)

    # Pytest doesn't find some of these :/
    def test_forward_no_throw(self):
        super().test_forward_no_throw()

    def test_forward_shape(self):
        super().test_forward_shape()

    def test_forward_value(self):
        super().test_forward_value()


class TestGradients:
    regress_on_current_results: bool = REGRESS_ON_CURRENT_RESULTS
    mlp_grads_path: Path = Path(__file__).parent / "test_data" / "mlp_grads.safetensors"
    siren_grads_path: Path = Path(__file__).parent / "test_data" / "siren_grads.safetensors"
    mlp_laplacians_path: Path = Path(__file__).parent / "test_data" / "mlp_laplacians.safetensors"
    siren_laplacians_path: Path = Path(__file__).parent / "test_data" / "siren_laplacians.safetensors"
    mlp_divergences_path: Path = Path(__file__).parent / "test_data" / "mlp_divergences.safetensors"
    siren_divergences_path: Path = Path(__file__).parent / "test_data" / "siren_divergences.safetensors"

    @pytest.fixture(autouse=True)
    def setup(self):
        # Create non-random models
        set_seed(0)
        self.mlp = MLP(in_features=1, out_features=1, hidden_features=10, hidden_layers=3, bias=True) # fmt: skip
        self.siren = SIREN(in_features=1, out_features=1, hidden_features=10, hidden_layers=2, bias=True, last_layer_linear=True, first_omega_0=20, hidden_omega_0=20) # fmt: skip
        set_seed(0)
        for p in self.mlp.parameters():
            nn.init.normal_(p, mean=0.0, std=0.1)
        for p in self.siren.parameters():
            nn.init.normal_(p, mean=0.0, std=0.1)
        # Create non-random inputs
        # NOTE: strided across all models, so each even alue is for mlp and each odd is for siren
        inputs = HelperTestForward.generate_realistic_inputs(0, [self.mlp, self.siren])
        assert len(inputs) % (2*4) == 0, "Need even number of inputs" # fmt: skip
        self.mlp_inputs = inputs[::2]
        self.siren_inputs = inputs[1::2]
        assert len(self.mlp_inputs) == len(self.siren_inputs), "Need even split of inputs" # fmt: skip
        # Save them if necessary an d well as their outputs
        if self.regress_on_current_results:
            # 100 tess per each of the functions using random input and some zero inputs
            mlp_outputs_coords = [self.mlp(input) for input in self.mlp_inputs] # fmt: skip
            siren_outputs_coords = [self.siren(input) for input in self.siren_inputs] # fmt: skip

            # Generate the stuff will save
            self.mlp_grads = [gradient(output, input) for output, input in mlp_outputs_coords] # fmt: skip
            self.siren_grads = [gradient(output, input) for output, input in siren_outputs_coords] # fmt: skip
            self.mlp_laplacians = [laplace(output, input) for output, input in mlp_outputs_coords] # fmt: skip
            self.siren_laplacians = [laplace(output, input) for output, input in siren_outputs_coords] # fmt: skip
            self.mlp_divergences = [divergence(output, input) for output, input in mlp_outputs_coords] # fmt: skip
            self.siren_divergences = [divergence(output, input) for output, input in siren_outputs_coords] # fmt: skip

            # Save it
            save_file(list2dict(self.mlp_grads), self.mlp_grads_path)
            save_file(list2dict(self.siren_grads), self.siren_grads_path)
            save_file(list2dict(self.mlp_laplacians), self.mlp_laplacians_path)
            save_file(list2dict(self.siren_laplacians), self.siren_laplacians_path)
            save_file(list2dict(self.mlp_divergences), self.mlp_divergences_path)
            save_file(list2dict(self.siren_divergences), self.siren_divergences_path) 
        # Load it
        self.mlp_grads = dict2list(load_file(self.mlp_grads_path)) # fmt: skip
        self.siren_grads = dict2list(load_file(self.siren_grads_path)) # fmt: skip
        self.mlp_laplacians = dict2list(load_file(self.mlp_laplacians_path)) # fmt: skip
        self.siren_laplacians = dict2list(load_file(self.siren_laplacians_path)) # fmt: skip
        self.mlp_divergences = dict2list(load_file(self.mlp_divergences_path)) # fmt: skip
        self.siren_divergences = dict2list(load_file(self.siren_divergences_path)) # fmt: skip


    def helper_loop(self, fn, name: str, expectss: Tuple[List[torch.Tensor], List[torch.Tensor]]):
        assert len(expectss) == 2, "Need two lists of expects" # fmt: skip
        for model, inputs, expects in list(zip(
            [self.mlp, self.siren],
            [self.mlp_inputs, self.siren_inputs], 
            expectss
        )):
            for input, expect in list(zip(inputs, expects)):
                model_output, coords = model(input)
                z = fn(model_output, coords)
                mxae = torch.abs(z - expect).max().item()
                # NOTE: use a more generous atol since this is a longer computational graph
                # we might be operating with and there can be compounding error
                assert torch.allclose(z, expect, atol=1e-4), f"{name} mismatch (model={model.__class__.__name__}); mxae={mxae:9e}" # fmt: skip

    # NOTE all use y, x order of inputs so we can just plug in like this
    # To students: if you read this, note that you might use the functions slightly differently irl
    def test_gradient(self):
        self.helper_loop(gradient, "gradient", [self.mlp_grads, self.siren_grads])

    def test_laplacian(self):
        self.helper_loop(laplace, "laplacian", [self.mlp_laplacians, self.siren_laplacians])

    def test_divergence(self):
        self.helper_loop(divergence, "divergence", [self.mlp_divergences, self.siren_divergences])


################################ MC FORMAT TESTERS ################################
class Question(pydantic.BaseModel):
    CorrectAnswers: Optional[List[str]] = None  # List of correct answers
    Answers: Optional[List[str]] = None  # Student answers
    Options: Dict[str, str] | List[str] | None = None  # List of possible answers
    Question: str  # Not used lmao


class MultipleChoice(pydantic.BaseModel):
    Questions: List[Question]


def test_multiple_choice_format():
    yaml_path = Path(__file__).parent / "multiple_choice.yml"
    assert yaml_path.exists(), "multiple_choice.yml does not exist"

    # Make sure the format is parseable
    contents = parse_yaml_file_as(MultipleChoice, yaml_path)
    assert len(contents.Questions) == 5, "multiple_choice.yml should have 5 qs)"

    # Make sure the formats look ok
    assert all(
        q.CorrectAnswers is None for q in contents.Questions
    ), "All CorrectAnswers should be None"
    assert len(set(q.Question for q in contents.Questions)) == len(
        contents.Questions
    ), "All questions should be unique"
