import asyncio

from ingestion.deriv_client import DerivClient


def _client() -> DerivClient:
    return DerivClient(app_id="1", api_token="tok", ws_url="wss://x", options_token_url="https://x")


def test_route_contract_update_delivers_to_the_matching_queue():
    client = _client()
    queue = asyncio.Queue()
    client._contract_queues[42] = queue

    client._route_contract_update({
        "proposal_open_contract": {"contract_id": 42, "is_sold": 0},
        "subscription": {"id": "sub-abc"},
    })

    assert queue.qsize() == 1
    assert client._contract_subscription_ids[42] == "sub-abc"


def test_route_contract_update_ignores_unmatched_contract_id():
    client = _client()
    # no queue registered for contract_id 7 -- must not raise
    client._route_contract_update({"proposal_open_contract": {"contract_id": 7, "is_sold": 0}})


def test_route_contract_update_drops_oldest_when_queue_full():
    client = _client()
    queue = asyncio.Queue(maxsize=1)
    client._contract_queues[1] = queue

    client._route_contract_update({"proposal_open_contract": {"contract_id": 1, "is_sold": 0, "tick": 1}})
    client._route_contract_update({"proposal_open_contract": {"contract_id": 1, "is_sold": 0, "tick": 2}})

    assert queue.qsize() == 1
    assert queue.get_nowait()["tick"] == 2  # newest kept, oldest dropped


def test_wait_for_contract_settlement_subscribes_once_then_consumes_pushes():
    async def run():
        client = _client()
        send_calls = []

        async def fake_send(payload):
            send_calls.append(payload)
            if "proposal_open_contract" in payload and payload.get("subscribe") == 1:
                return {"proposal_open_contract": {"contract_id": 1, "is_sold": 0},
                         "subscription": {"id": "sub-1"}}
            if "forget" in payload:
                return {"forget": 1}
            raise AssertionError(f"unexpected payload: {payload}")

        client._send = fake_send

        async def push_updates():
            await asyncio.sleep(0.01)
            client._route_contract_update({"proposal_open_contract": {"contract_id": 1, "is_sold": 0}})
            await asyncio.sleep(0.01)
            client._route_contract_update({"proposal_open_contract": {"contract_id": 1, "is_sold": 1, "profit": 3.0}})

        pusher = asyncio.create_task(push_updates())
        contract = await client.wait_for_contract_settlement(1, timeout=5.0)
        await pusher

        assert contract["is_sold"] == 1
        assert contract["profit"] == 3.0
        # exactly one subscribe request + one forget -- never polled
        subscribe_calls = [p for p in send_calls if p.get("subscribe") == 1]
        forget_calls = [p for p in send_calls if "forget" in p]
        assert len(subscribe_calls) == 1
        assert len(forget_calls) == 1
        # cleaned up after settling
        assert 1 not in client._contract_queues
        assert 1 not in client._contract_subscription_ids

    asyncio.run(run())


def test_wait_for_contract_settlement_times_out_and_still_unsubscribes():
    async def run():
        client = _client()
        forgotten = []

        async def fake_send(payload):
            if payload.get("subscribe") == 1:
                return {"proposal_open_contract": {"contract_id": 1, "is_sold": 0},
                         "subscription": {"id": "sub-1"}}
            if "forget" in payload:
                forgotten.append(payload["forget"])
                return {"forget": 1}
            raise AssertionError(f"unexpected payload: {payload}")

        client._send = fake_send
        contract = await client.wait_for_contract_settlement(1, timeout=0.05)

        assert not contract.get("is_sold")
        assert forgotten == ["sub-1"]  # still unsubscribed even on timeout

    asyncio.run(run())


def test_proposal_open_contract_is_rate_limited_key():
    client = _client()
    assert "proposal_open_contract" in client._rate_limited_keys
