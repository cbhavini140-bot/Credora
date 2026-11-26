"""Comprehensive backend & end-to-end workflow tests for CREDORA."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _auth_header(client: TestClient, email: str, password: str) -> dict[str, str]:
    resp = client.post("/api/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_and_three_models_loaded(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    for key in ("ulb", "sparkov", "paysim"):
        assert data["models"][key]["loaded"] is True, f"Model {key} failed: {data['models'][key]}"


def test_auth_logout_invalidates_token_and_rbac(client: TestClient):
    analyst_hdr = _auth_header(client, "analyst@credora.dev", "CredoraAnalyst!2026")
    # Analyst cannot start simulation or update user roles (ADMIN only)
    sim_resp = client.post("/api/simulation/start", headers=analyst_hdr)
    assert sim_resp.status_code == 403

    role_resp = client.patch(
        "/api/auth/users/1/role",
        json={"role": "ANALYST"},
        headers=analyst_hdr,
    )
    assert role_resp.status_code == 403

    # Logout invalidates token
    logout_resp = client.post("/api/auth/logout", headers=analyst_hdr)
    assert logout_resp.status_code == 200
    me_after = client.get("/api/auth/me", headers=analyst_hdr)
    assert me_after.status_code == 401


def test_single_predictions_and_shap_all_models(client: TestClient):
    hdr = _auth_header(client, "analyst@credora.dev", "CredoraAnalyst!2026")

    for model_key in ("ulb", "sparkov", "paysim"):
        schema_resp = client.get(f"/api/predictions/sample-schema/{model_key}", headers=hdr)
        assert schema_resp.status_code == 200
        example = schema_resp.json()["example_fraud"] or schema_resp.json()["example_legit"]
        assert example, f"Missing sample example for {model_key}"

        pred_resp = client.post(
            "/api/predictions/single",
            json={"model_key": model_key, "features": example, "save_transaction": True, "include_shap": True},
            headers=hdr,
        )
        assert pred_resp.status_code == 200, pred_resp.text
        pred = pred_resp.json()["prediction"]
        assert 0.0 <= pred["fraud_probability"] <= 1.0
        assert 0.0 <= pred["risk_score"] <= 100.0
        assert pred["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert len(pred["shap_explanation"]["top_contributions"]) > 0

        if model_key == "ulb":
            assert "PCA" in pred["shap_explanation"]["explainability_note"]


def test_csv_validation_rejects_mismatched_schema(client: TestClient):
    hdr = _auth_header(client, "analyst@credora.dev", "CredoraAnalyst!2026")
    bad_csv = b"col_a,col_b\n1,2\n"
    resp = client.post(
        "/api/predictions/batch",
        data={"model_key": "paysim"},
        files={"file": ("invalid.csv", bad_csv, "text/csv")},
        headers=hdr,
    )
    assert resp.status_code == 422
    assert "schema mismatch" in resp.json()["detail"].lower()


def test_end_to_end_acceptance_workflow(client: TestClient):
    # 1. Log in as fraud analyst
    hdr = _auth_header(client, "analyst@credora.dev", "CredoraAnalyst!2026")
    me = client.get("/api/auth/me", headers=hdr).json()
    analyst_id = me["id"]

    # Baseline analytics
    before_stats = client.get("/api/analytics/overview", headers=hdr).json()["kpis"]

    # 2-7. Upload valid PaySim CSV, select paysim model, process, save, generate alerts
    sample_path = get_settings().datasets_path / "samples" / "paysim_sample.csv"
    assert sample_path.exists()
    with sample_path.open("rb") as fh:
        batch_resp = client.post(
            "/api/predictions/batch",
            data={"model_key": "paysim"},
            files={"file": ("paysim_sample.csv", fh.read(), "text/csv")},
            headers=hdr,
        )
    assert batch_resp.status_code == 200, batch_resp.text
    batch_data = batch_resp.json()
    assert batch_data["total_processed"] == 100
    assert batch_data["predicted_fraud_count"] > 0
    assert batch_data["alerts_generated"] > 0

    # Verify CSV export
    export_resp = client.get(batch_data["download_url"], headers=hdr)
    assert export_resp.status_code == 200
    assert "transaction_ref" in export_resp.text

    # 8. Open an alert
    alerts_resp = client.get("/api/alerts?status=OPEN", headers=hdr)
    assert alerts_resp.status_code == 200
    alerts_items = alerts_resp.json()["items"]
    assert len(alerts_items) > 0
    target_alert = alerts_items[0]

    detail_resp = client.get(f"/api/alerts/{target_alert['id']}", headers=hdr)
    assert detail_resp.status_code == 200

    # 9-12. Create investigation, add note, assign investigation, record final analyst decision
    inv_resp = client.post(
        "/api/investigations",
        json={
            "title": "E2E PaySim High-Risk Transfer Review",
            "summary": "Reviewing flagged transfer anomaly from batch CSV upload.",
            "priority": "CRITICAL",
            "transaction_ids": [target_alert["transaction_id"]],
        },
        headers=hdr,
    )
    assert inv_resp.status_code == 201, inv_resp.text
    inv = inv_resp.json()

    note_resp = client.post(
        f"/api/investigations/{inv['id']}/notes",
        json={"content": "Verified origin balance depletion anomaly via SHAP explanation."},
        headers=hdr,
    )
    assert note_resp.status_code == 201

    assign_resp = client.patch(
        f"/api/investigations/{inv['id']}",
        json={"assigned_to_id": analyst_id, "status": "IN_PROGRESS"},
        headers=hdr,
    )
    assert assign_resp.status_code == 200
    assert assign_resp.json()["assigned_to_id"] == analyst_id

    decision_resp = client.patch(
        f"/api/investigations/{inv['id']}",
        json={"final_decision": "CONFIRMED_FRAUD"},
        headers=hdr,
    )
    assert decision_resp.status_code == 200
    assert decision_resp.json()["final_decision"] == "CONFIRMED_FRAUD"
    assert decision_resp.json()["status"] == "CLOSED_FRAUD"

    # 13. Verify dashboard statistics updated
    after_stats = client.get("/api/analytics/overview", headers=hdr).json()["kpis"]
    assert after_stats["total_transactions"] >= before_stats["total_transactions"] + 100
    assert after_stats["confirmed_fraud"] >= before_stats["confirmed_fraud"] + 1

    # 14. Verify audit log records actions
    audit_resp = client.get("/api/audit?page_size=50", headers=hdr)
    assert audit_resp.status_code == 200
    actions = {item["action"] for item in audit_resp.json()["items"]}
    for expected_action in ("LOGIN", "BATCH_UPLOAD", "INVESTIGATION_CREATED", "NOTE_ADDED", "ANALYST_DECISION"):
        assert expected_action in actions
