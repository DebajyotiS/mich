"""Tests for gradient and derivative functions in torch_utils."""

import torch

from mich.utils.torch_utils import _softplus_deriv


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


class TestGradientConsistency:
    """Tests for consistency across different derivative functions."""

    def test_all_derivatives_have_correct_shape(self):
        """All derivative functions should preserve input shape."""
        x = torch.randn(3, 4, 5)

        assert _softplus_deriv(x).shape == x.shape

    def test_batch_processing(self):
        """Functions should work correctly with batch inputs."""
        batch_size = 32
        x = torch.randn(batch_size)

        soft_deriv = _softplus_deriv(x)

        assert soft_deriv.shape == (batch_size,)

    def test_gradient_computation(self):
        """Test that functions work with autograd."""
        x = torch.randn(5, requires_grad=True)

        # This should not raise errors when computing gradients
        y2 = _softplus_deriv(x).sum()

        y2.backward(retain_graph=True)

        assert x.grad is not None


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_very_large_positive_values(self):
        """Functions should handle very large values gracefully."""
        x = torch.tensor([100.0, 1000.0])

        assert torch.isfinite(_softplus_deriv(x)).all()

    def test_very_large_negative_values(self):
        """Functions should handle very large negative values gracefully."""
        x = torch.tensor([-100.0, -1000.0])

        assert torch.isfinite(_softplus_deriv(x)).all()

    def test_zero_input(self):
        """Functions should work correctly at x=0."""
        x = torch.zeros(5)

        assert torch.isfinite(_softplus_deriv(x)).all()

    def test_different_dtypes(self):
        """Functions should work with different tensor dtypes."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(3, 3, dtype=dtype)

            result1 = _softplus_deriv(x)

            assert result1.dtype == dtype
