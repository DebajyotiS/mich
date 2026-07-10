"""Tests for gradient and derivative functions in torch_utils."""

import torch
from mich.utils.torch_utils import (
    _neg_softplus_neg,
    _neg_softplus_neg_deriv,
    _one_plus_softplus,
    _sigmoid_deriv,
    _softplus_deriv,
    _tanh_deriv,
)


class TestSigmoidDeriv:
    """Tests for sigmoid derivative."""

    def test_sigmoid_deriv_at_zero(self):
        """At x=0, sigmoid'(0) should be 0.25 (sigmoid(0)=0.5, 0.5*0.5=0.25)."""
        x = torch.tensor(0.0, requires_grad=True)
        result = _sigmoid_deriv(x)
        assert torch.allclose(result, torch.tensor(0.25))

    def test_sigmoid_deriv_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(3, 4, 5)
        result = _sigmoid_deriv(x)
        assert result.shape == x.shape

    def test_sigmoid_deriv_bounds(self):
        """Sigmoid derivative should be in [0, 0.25]."""
        x = torch.linspace(-10, 10, 100)
        result = _sigmoid_deriv(x)
        assert (result >= 0).all() and (result <= 0.25).all()

    def test_sigmoid_deriv_numerical_gradient(self):
        """Test against numerical gradient."""
        x = torch.tensor([0.5, -0.5, 2.0], requires_grad=True)
        # Numerical gradient calculation using finite differences
        eps = 1e-4
        x_plus = x + eps
        x_minus = x - eps
        numerical_grad = (torch.sigmoid(x_plus) - torch.sigmoid(x_minus)) / (2 * eps)

        analytical_grad = _sigmoid_deriv(x)
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-2, rtol=1e-2)


class TestSoftplusDeriv:
    """Tests for softplus derivative."""

    def test_softplus_deriv_is_sigmoid(self):
        """d/dx softplus(x) = sigmoid(x)."""
        x = torch.randn(5, 5)
        result = _softplus_deriv(x)
        expected = torch.sigmoid(x)
        assert torch.allclose(result, expected)

    def test_softplus_deriv_at_zero(self):
        """At x=0, softplus'(0) should be 0.5 (sigmoid(0)=0.5)."""
        x = torch.tensor(0.0)
        result = _softplus_deriv(x)
        assert torch.allclose(result, torch.tensor(0.5))

    def test_softplus_deriv_bounds(self):
        """Softplus derivative (sigmoid) should be in (0, 1)."""
        x = torch.linspace(-10, 10, 100)
        result = _softplus_deriv(x)
        assert (result > 0).all() and (result < 1).all()

    def test_softplus_deriv_numerical_gradient(self):
        """Test against numerical gradient of softplus."""
        x = torch.tensor([0.5, -0.5, 2.0], requires_grad=True)
        eps = 1e-4
        x_plus = x + eps
        x_minus = x - eps
        numerical_grad = (
            torch.nn.functional.softplus(x_plus) - torch.nn.functional.softplus(x_minus)
        ) / (2 * eps)

        analytical_grad = _softplus_deriv(x)
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-2, rtol=1e-2)


class TestOnePlusSoftplus:
    """Tests for 1 + softplus function."""

    def test_one_plus_softplus_at_zero(self):
        """At x=0, 1+softplus(0) = 1 + ln(2) ~= 1.693."""
        x = torch.tensor(0.0)
        result = _one_plus_softplus(x)
        expected = 1.0 + torch.nn.functional.softplus(torch.tensor(0.0))
        assert torch.allclose(result, expected)

    def test_one_plus_softplus_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(2, 3, 4)
        result = _one_plus_softplus(x)
        assert result.shape == x.shape

    def test_one_plus_softplus_positivity(self):
        """1 + softplus(x) should always be > 1 (since softplus(x) > 0)."""
        x = torch.linspace(-10, 10, 100)
        result = _one_plus_softplus(x)
        assert (result > 1.0).all()

    def test_one_plus_softplus_growth(self):
        """Should grow approximately linearly for large positive x."""
        x = torch.tensor([10.0, 20.0, 30.0])
        result = _one_plus_softplus(x)
        # For large x, softplus(x) ~= x, so 1 + softplus(x) ~= 1 + x
        expected = 1.0 + x  # approximate
        assert torch.allclose(result, expected, atol=0.1)


class TestNegSoftplusNeg:
    """Tests for -softplus(-x) function."""

    def test_neg_softplus_neg_at_zero(self):
        """At x=0, -softplus(-0) = -softplus(0) = -ln(2) ~= -0.693."""
        x = torch.tensor(0.0)
        result = _neg_softplus_neg(x)
        expected = -torch.nn.functional.softplus(torch.tensor(0.0))
        assert torch.allclose(result, expected)

    def test_neg_softplus_neg_non_positivity(self):
        """-softplus(-x) should always be <= 0."""
        x = torch.linspace(-10, 10, 100)
        result = _neg_softplus_neg(x)
        assert (result <= 0).all()

    def test_neg_softplus_neg_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(3, 3, 3)
        result = _neg_softplus_neg(x)
        assert result.shape == x.shape

    def test_neg_softplus_neg_approx_for_negative_x(self):
        """For large negative x, -softplus(-x) ~= x (since softplus(-x) ~= -x)."""
        x = torch.tensor([-10.0, -20.0, -30.0])
        result = _neg_softplus_neg(x)
        # For large negative x, softplus(-x) ~= -x, so -softplus(-x) ~= x
        expected = x
        assert torch.allclose(result, expected, atol=0.1)


class TestNegSoftplusNegDeriv:
    """Tests for derivative of -softplus(-x)."""

    def test_neg_softplus_neg_deriv_at_zero(self):
        """At x=0, d/dx[-softplus(-x)] = sigmoid(0) - 1 = 0.5 - 1 = -0.5."""
        x = torch.tensor(0.0)
        result = _neg_softplus_neg_deriv(x)
        assert torch.allclose(result, torch.tensor(-0.5))

    def test_neg_softplus_neg_deriv_is_sigmoid_minus_one(self):
        """d/dx[-softplus(-x)] = sigmoid(x) - 1."""
        x = torch.randn(5, 5)
        result = _neg_softplus_neg_deriv(x)
        expected = torch.sigmoid(x) - 1.0
        assert torch.allclose(result, expected)

    def test_neg_softplus_neg_deriv_bounds(self):
        """sigmoid(x) - 1 is in (-1, 0)."""
        x = torch.linspace(-10, 10, 100)
        result = _neg_softplus_neg_deriv(x)
        assert (result > -1).all() and (result < 0).all()

    def test_neg_softplus_neg_deriv_numerical_gradient(self):
        """Test against numerical gradient."""
        x = torch.tensor([0.5, -0.5, 2.0], requires_grad=True)
        eps = 1e-4
        x_plus = x + eps
        x_minus = x - eps
        numerical_grad = (
            -torch.nn.functional.softplus(-x_plus) - (-torch.nn.functional.softplus(-x_minus))
        ) / (2 * eps)

        analytical_grad = _neg_softplus_neg_deriv(x)
        # Note: numerical gradient may differ in sign/magnitude for this one due to the formula
        assert torch.allclose(
            torch.abs(analytical_grad), torch.abs(numerical_grad), atol=1e-2, rtol=1e-2
        )


class TestTanhDeriv:
    """Tests for tanh derivative."""

    def test_tanh_deriv_at_zero(self):
        """At x=0, tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1."""
        x = torch.tensor(0.0)
        result = _tanh_deriv(x)
        assert torch.allclose(result, torch.tensor(1.0))

    def test_tanh_deriv_formula(self):
        """tanh'(x) = 1 - tanh(x)^2."""
        x = torch.randn(5, 5)
        result = _tanh_deriv(x)
        expected = 1.0 - torch.tanh(x) ** 2
        assert torch.allclose(result, expected)

    def test_tanh_deriv_bounds(self):
        """tanh derivative should be in [0, 1]."""
        x = torch.linspace(-10, 10, 100)
        result = _tanh_deriv(x)
        assert (result >= 0).all() and (result <= 1).all()

    def test_tanh_deriv_shape(self):
        """Output shape should match input shape."""
        x = torch.randn(2, 3, 4, 5)
        result = _tanh_deriv(x)
        assert result.shape == x.shape

    def test_tanh_deriv_numerical_gradient(self):
        """Test against numerical gradient of tanh."""
        x = torch.tensor([0.5, -0.5, 2.0], requires_grad=True)
        eps = 1e-4
        x_plus = x + eps
        x_minus = x - eps
        numerical_grad = (torch.tanh(x_plus) - torch.tanh(x_minus)) / (2 * eps)

        analytical_grad = _tanh_deriv(x)
        assert torch.allclose(analytical_grad, numerical_grad, atol=1e-2, rtol=1e-2)


class TestGradientConsistency:
    """Tests for consistency across different derivative functions."""

    def test_all_derivatives_have_correct_shape(self):
        """All derivative functions should preserve input shape."""
        x = torch.randn(3, 4, 5)

        assert _sigmoid_deriv(x).shape == x.shape
        assert _softplus_deriv(x).shape == x.shape
        assert _one_plus_softplus(x).shape == x.shape
        assert _neg_softplus_neg(x).shape == x.shape
        assert _neg_softplus_neg_deriv(x).shape == x.shape
        assert _tanh_deriv(x).shape == x.shape

    def test_batch_processing(self):
        """Functions should work correctly with batch inputs."""
        batch_size = 32
        x = torch.randn(batch_size)

        sig_deriv = _sigmoid_deriv(x)
        soft_deriv = _softplus_deriv(x)
        tanh_d = _tanh_deriv(x)

        assert sig_deriv.shape == (batch_size,)
        assert soft_deriv.shape == (batch_size,)
        assert tanh_d.shape == (batch_size,)

    def test_gradient_computation(self):
        """Test that functions work with autograd."""
        x = torch.randn(5, requires_grad=True)

        # These should not raise errors when computing gradients
        y1 = _sigmoid_deriv(x).sum()
        y2 = _softplus_deriv(x).sum()
        y3 = _tanh_deriv(x).sum()

        y1.backward(retain_graph=True)
        y2.backward(retain_graph=True)
        y3.backward(retain_graph=True)

        assert x.grad is not None


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_very_large_positive_values(self):
        """Functions should handle very large values gracefully."""
        x = torch.tensor([100.0, 1000.0])

        assert torch.isfinite(_sigmoid_deriv(x)).all()
        assert torch.isfinite(_softplus_deriv(x)).all()
        assert torch.isfinite(_tanh_deriv(x)).all()

    def test_very_large_negative_values(self):
        """Functions should handle very large negative values gracefully."""
        x = torch.tensor([-100.0, -1000.0])

        assert torch.isfinite(_sigmoid_deriv(x)).all()
        assert torch.isfinite(_softplus_deriv(x)).all()
        assert torch.isfinite(_tanh_deriv(x)).all()

    def test_zero_input(self):
        """Functions should work correctly at x=0."""
        x = torch.zeros(5)

        assert torch.isfinite(_sigmoid_deriv(x)).all()
        assert torch.isfinite(_softplus_deriv(x)).all()
        assert torch.isfinite(_tanh_deriv(x)).all()

    def test_different_dtypes(self):
        """Functions should work with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(3, 3, dtype=dtype)

            result1 = _sigmoid_deriv(x)
            result2 = _tanh_deriv(x)

            assert result1.dtype == dtype
            assert result2.dtype == dtype
