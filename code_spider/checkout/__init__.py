"""Repository checkout (local path or shallow git clone) keyed by commit SHA."""

from code_spider.checkout.git import CheckoutResult, ensure_checkout

__all__ = ["CheckoutResult", "ensure_checkout"]
