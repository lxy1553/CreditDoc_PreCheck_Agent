import json
from io import BytesIO
from unittest.mock import patch

import pytest

from creditdoc_precheck_agent.document_parser import DocumentParseError, PrivateDocumentParser


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_private_parser_returns_page_provenance_and_confident_fields():
    payload = {"document_type": "income_proof", "pages": [{"page": 1, "text": "模拟文本"}],
               "fields": {"monthly_income": {"value": "9000 元", "confidence": 0.97, "page": 1}}}
    parser = PrivateDocumentParser("http://127.0.0.1:9999", "parser-v1")
    with patch("creditdoc_precheck_agent.document_parser.open_without_redirect",
               return_value=FakeResponse(json.dumps(payload).encode())) as opener:
        result = parser.parse(b"%PDF-demo", filename="fake.pdf", content_type="application/pdf")
    assert result["document_type"] == "income_proof"
    assert result["fields"]["monthly_income"]["page"] == 1
    request = opener.call_args.args[0]
    assert request.full_url.endswith("/v1/parse")


def test_private_parser_rejects_unknown_field_page_reference():
    payload = {"document_type": "income_proof", "pages": [{"page": 1, "text": ""}],
               "fields": {"monthly_income": {"value": "9000", "confidence": 0.9, "page": 2}}}
    parser = PrivateDocumentParser("http://127.0.0.1:9999", "parser-v1")
    with patch("creditdoc_precheck_agent.document_parser.open_without_redirect",
               return_value=FakeResponse(json.dumps(payload).encode())):
        with pytest.raises(DocumentParseError, match="page"):
            parser.parse(b"%PDF-demo", filename="fake.pdf", content_type="application/pdf")


def test_private_parser_rejects_oversized_and_non_document_uploads():
    parser = PrivateDocumentParser("http://127.0.0.1:9999", "parser-v1", max_bytes=3)
    with pytest.raises(DocumentParseError, match="size"):
        parser.parse(b"1234", filename="fake.pdf", content_type="application/pdf")
    with pytest.raises(DocumentParseError, match="supported"):
        parser.parse(b"x", filename="fake.exe", content_type="application/octet-stream")


def test_private_parser_rejects_public_endpoint_without_allowlist():
    with pytest.raises(ValueError, match="private or explicitly allowlisted"):
        PrivateDocumentParser("https://parser.example.com", "parser-v1")


def test_simulated_parser_only_returns_explicit_fictional_demo_cases():
    from creditdoc_precheck_agent.document_parser import SimulatedDocumentParser

    parser = SimulatedDocumentParser(environment="synthetic-demo")
    result = parser.parse(b"not a real applicant file", filename="demo-income-proof.pdf", content_type="application/pdf")
    assert result["document_type"] == "income_proof"
    assert result["fields"]["monthly_income"]["value"] == "10000 元"
    with pytest.raises(DocumentParseError, match="demo filename"):
        parser.parse(b"x", filename="customer-upload.pdf", content_type="application/pdf")
    with pytest.raises(ValueError, match="synthetic demo"):
        SimulatedDocumentParser(environment="approved-knowledge")


def test_simulated_parser_supports_low_confidence_and_matching_income_scenarios():
    from creditdoc_precheck_agent.document_parser import SimulatedDocumentParser

    parser = SimulatedDocumentParser(environment="synthetic-demo")
    low_confidence = parser.parse(b"placeholder", filename="demo-application-blurry.pdf",
                                  content_type="application/pdf")
    matched_income = parser.parse(b"placeholder", filename="demo-income-proof-matching.pdf",
                                  content_type="application/pdf")
    assert low_confidence["fields"]["applicant_name"]["confidence"] == 0.62
    assert matched_income["fields"]["monthly_income"]["value"] == "12000 元"


def test_private_parser_rejects_boolean_page_number():
    payload = {"document_type": "income_proof", "pages": [{"page": True, "text": ""}], "fields": {}}
    with pytest.raises(DocumentParseError, match="page"):
        PrivateDocumentParser._validate(payload)
