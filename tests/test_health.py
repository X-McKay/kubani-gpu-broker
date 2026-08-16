async def test_healthz(broker_client):
    resp = await broker_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.text == "ok"


async def test_readyz(broker_client):
    resp = await broker_client.get("/readyz")
    assert resp.status_code == 200


async def test_metrics_endpoint_serves_prometheus(broker_client):
    resp = await broker_client.get("/metrics")
    assert resp.status_code == 200
    assert "kubani_gpu_broker" in resp.text
