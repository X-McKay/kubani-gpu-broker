async def test_healthz(broker_client):
    resp = await broker_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.text == "ok"


async def test_readyz(broker_client):
    resp = await broker_client.get("/readyz")
    assert resp.status_code == 200


async def test_public_app_does_not_serve_metrics(broker_client):
    # Metrics live on the admin listener only (spec section 24.2).
    resp = await broker_client.get("/metrics")
    assert resp.status_code == 404
