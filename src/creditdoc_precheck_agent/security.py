"""Small shared network-boundary helpers for private model adapters."""

from __future__ import annotations

import ipaddress
import urllib.request


def is_private_or_loopback(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized in {"localhost", "localhost.localdomain"} or normalized.endswith((".local", ".internal", ".intranet", ".lan")):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_private or address.is_loopback


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def open_without_redirect(request, timeout: float):
    """Send a request without following redirects to another host."""
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(request, timeout=timeout)
