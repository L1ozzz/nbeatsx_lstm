"""Run opposite-order Sepolia comparisons of raw bytes and packed bytes32."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from web3 import Web3

from packing import assert_round_trip, format_text_record, parse_text_record, unpack_record


CHAIN_ID = 11155111
READ_REPETITIONS_PER_CONDITION_PER_DEPLOYMENT = 30
HERE = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hex_value(value) -> str:
    if isinstance(value, str):
        return value
    result = value.hex()
    return result if result.startswith("0x") else "0x" + result


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes) or hasattr(value, "hex"):
        try:
            return hex_value(value)
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def load_credentials(script_path: Path) -> tuple[str, str]:
    """Read legacy local secrets at runtime without writing them to evidence."""

    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    private_key = None
    rpc_url = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "private_key":
            private_key = ast.literal_eval(node.value)
        elif target.id == "w3":
            for child in ast.walk(node.value):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if child.value.startswith(("https://", "http://")):
                        rpc_url = child.value
                        break
    if not private_key or not rpc_url:
        raise RuntimeError("Could not load the existing local RPC URL/private key")
    return rpc_url, private_key


def make_web3(url: str) -> Web3:
    session = requests.Session()
    session.trust_env = False
    return Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 60}, session=session))


def fee_fields(w3: Web3) -> dict[str, int]:
    block = w3.eth.get_block("latest")
    base_fee = int(block.get("baseFeePerGas") or w3.eth.gas_price)
    priority = w3.to_wei(2, "gwei")
    return {"maxPriorityFeePerGas": priority, "maxFeePerGas": base_fee * 2 + priority}


def send(w3: Web3, account, tx: dict, label: str):
    estimate = int(w3.eth.estimate_gas(tx))
    tx["gas"] = max(estimate + 20_000, int(estimate * 1.20))
    started = time.perf_counter()
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"{label}: sent {hex_value(tx_hash)}", flush=True)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
    elapsed = time.perf_counter() - started
    print(
        f"{label}: status={receipt.status} block={receipt.blockNumber} "
        f"gas={receipt.gasUsed} elapsed={elapsed:.3f}s",
        flush=True,
    )
    return receipt, estimate, elapsed


def calldata_analysis(data) -> dict[str, int]:
    payload = bytes(data)
    zeros = payload.count(0)
    nonzeros = len(payload) - zeros
    return {
        "bytes": len(payload),
        "zero_bytes": zeros,
        "nonzero_bytes": nonzeros,
        "intrinsic_plus_calldata_gas": 21_000 + zeros * 4 + nonzeros * 16,
    }


def transaction_proof(w3: Web3, receipt) -> dict:
    tx = w3.eth.get_transaction(receipt.transactionHash)
    return {
        "transaction": {
            "hash": hex_value(tx.hash),
            "block_number": tx.blockNumber,
            "from": tx["from"],
            "to": tx.to,
            "nonce": tx.nonce,
            "gas_limit": tx.gas,
            "gas_price": tx.gasPrice,
            "value": tx.value,
            "input": hex_value(tx.input),
        },
        "receipt": json_safe(dict(receipt)),
        "calldata": calldata_analysis(tx.input),
    }


def condition_objects(contract, date: int, raw: bytes, packed: bytes, condition: str):
    if condition == "raw":
        return {
            "payload": raw,
            "exists": contract.functions.rawDataExists(date),
            "store": contract.functions.storeRawData(date, raw),
            "get": contract.functions.getRawData(date),
            "event": contract.events.RawDataStored(),
        }
    if condition == "packed_bytes32":
        return {
            "payload": packed,
            "exists": contract.functions.packedDataExists(date),
            "store": contract.functions.storePackedData(date, packed),
            "get": contract.functions.getPackedData(date),
            "event": contract.events.PackedDataStored(),
        }
    raise ValueError(condition)


def read_once(call) -> tuple[bytes, float]:
    started = time.perf_counter()
    returned = bytes(call.call())
    return returned, time.perf_counter() - started


def run_deployment(w3: Web3, account, abi: list, bytecode: str, run: str, order: list[str], raw: bytes, packed: bytes, date: int):
    factory = w3.eth.contract(abi=abi, bytecode=bytecode)
    deploy_tx = factory.constructor().build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address, "pending"),
            "chainId": CHAIN_ID,
            **fee_fields(w3),
        }
    )
    deploy_receipt, deploy_estimate, deploy_elapsed = send(w3, account, deploy_tx, f"{run} deploy")
    if deploy_receipt.status != 1 or not deploy_receipt.contractAddress:
        raise RuntimeError(f"{run} deployment failed")
    address = Web3.to_checksum_address(deploy_receipt.contractAddress)
    code = bytes(w3.eth.get_code(address))
    contract = w3.eth.contract(address=address, abi=abi)
    owner = Web3.to_checksum_address(contract.functions.owner().call())
    if not code or owner != Web3.to_checksum_address(account.address):
        raise RuntimeError(f"{run} deployment verification failed")

    deployment = transaction_proof(w3, deploy_receipt)
    deployment.update(
        {
            "run": run,
            "contract_address": address,
            "estimated_gas": deploy_estimate,
            "elapsed_seconds": deploy_elapsed,
            "runtime_code_bytes": len(code),
            "runtime_code_sha256": sha256(code),
            "eth_getCode_nonempty": True,
            "owner": owner,
            "owner_matches_sender": True,
        }
    )
    (HERE / f"{run}_deployment.json").write_text(
        json.dumps(deployment, indent=2) + "\n", encoding="utf-8"
    )

    writes = []
    for sequence, condition in enumerate(order, start=1):
        obj = condition_objects(contract, date, raw, packed, condition)
        exists_before = bool(obj["exists"].call())
        if exists_before:
            raise RuntimeError(f"{run} {condition} was not a fresh mapping entry")
        tx = obj["store"].build_transaction(
            {
                "from": account.address,
                "nonce": w3.eth.get_transaction_count(account.address, "pending"),
                "chainId": CHAIN_ID,
                **fee_fields(w3),
            }
        )
        receipt, estimate, elapsed = send(w3, account, tx, f"{run} {condition}")
        events = obj["event"].process_receipt(receipt)
        returned = bytes(obj["get"].call()) if receipt.status == 1 else b""
        event_matches = (
            len(events) == 1
            and int(events[0]["args"]["date"]) == date
            and bytes(events[0]["args"]["data"]) == obj["payload"]
        )
        exists_after = bool(obj["exists"].call())
        roundtrip = returned == obj["payload"]
        decoded_exact = True
        if condition == "packed_bytes32":
            decoded_exact = format_text_record(unpack_record(returned)) == raw.decode("utf-8")
        passed = receipt.status == 1 and len(events) == 1 and event_matches and exists_after and roundtrip and decoded_exact
        if not passed:
            raise RuntimeError(f"{run} {condition} write/read verification failed")

        proof = transaction_proof(w3, receipt)
        proof.update(
            {
                "run": run,
                "sequence": sequence,
                "condition": condition,
                "date_key": date,
                "payload_bytes": len(obj["payload"]),
                "payload_hex": "0x" + obj["payload"].hex(),
                "payload_sha256": sha256(obj["payload"]),
                "estimated_write_gas": estimate,
                "confirmation_seconds": elapsed,
                "exists_before": exists_before,
                "receipt_status_ok": receipt.status == 1,
                "event_count": len(events),
                "event_matches_input": event_matches,
                "exists_after": exists_after,
                "get_by_date_matches_input": roundtrip,
                "packed_decode_matches_source_text": decoded_exact,
                "verification_passed": passed,
                "effective_gas_price_gwei": receipt.effectiveGasPrice / 1e9,
                "transaction_fee_eth": receipt.gasUsed * receipt.effectiveGasPrice / 1e18,
            }
        )
        (HERE / f"{run}_{condition}_write.json").write_text(
            json.dumps(proof, indent=2) + "\n", encoding="utf-8"
        )
        writes.append(proof)

    read_rows = []
    calls = {condition: condition_objects(contract, date, raw, packed, condition)["get"] for condition in ("raw", "packed_bytes32")}
    read_estimates = {condition: int(call.estimate_gas({"from": account.address})) for condition, call in calls.items()}
    for repetition in range(READ_REPETITIONS_PER_CONDITION_PER_DEPLOYMENT):
        pair_order = ("raw", "packed_bytes32") if repetition % 2 == 0 else ("packed_bytes32", "raw")
        for position, condition in enumerate(pair_order, start=1):
            returned, seconds = read_once(calls[condition])
            expected = raw if condition == "raw" else packed
            matches = returned == expected
            decoded_matches = True if condition == "raw" else format_text_record(unpack_record(returned)) == raw.decode("utf-8")
            if not matches or not decoded_matches:
                raise RuntimeError(f"{run} {condition} repeated read failed")
            read_rows.append(
                {
                    "run": run,
                    "contract_address": address,
                    "repetition": repetition + 1,
                    "pair_position": position,
                    "condition": condition,
                    "date_key": date,
                    "returned_bytes": len(returned),
                    "estimated_call_gas": read_estimates[condition],
                    "rpc_call_seconds": seconds,
                    "readback_matches": matches,
                    "decode_matches_source": decoded_matches,
                }
            )
    return {
        "run": run,
        "order": order,
        "contract_address": address,
        "deployment": deployment,
        "writes": writes,
        "read_estimates": read_estimates,
        "read_rows": read_rows,
    }


def timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
    return {
        "n": len(values),
        "mean_seconds": statistics.mean(values),
        "standard_deviation_seconds": statistics.stdev(values),
        "median_seconds": statistics.median(values),
        "q1_seconds": quartiles[0],
        "q3_seconds": quartiles[2],
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def main() -> None:
    credential_script = Path(
        os.environ.get(
            "ETH_STORE_SCRIPT",
            r"D:\Research-Haozhi Liao\NBeatsx-LSTM-副本\NBeatsx-LSTM\eth-store.py",
        )
    )
    artifact = json.loads((HERE / "contract_artifact.json").read_text(encoding="utf-8"))
    source_text = (HERE / "source_record.txt").read_text(encoding="utf-8").strip()
    record = parse_text_record(source_text)
    raw = source_text.encode("utf-8")
    packed = assert_round_trip(source_text)
    if len(raw) != 83 or len(packed) != 32:
        raise RuntimeError(f"Unexpected payload sizes raw={len(raw)} packed={len(packed)}")

    rpc_url, private_key = load_credentials(credential_script)
    w3 = make_web3(rpc_url)
    if not w3.is_connected() or w3.eth.chain_id != CHAIN_ID:
        raise RuntimeError("Could not connect to Sepolia")
    account = w3.eth.account.from_key(private_key)
    balance_before = w3.eth.get_balance(account.address)
    started = utc_now()

    runs = [
        run_deployment(
            w3,
            account,
            artifact["abi"],
            artifact["bytecode"],
            "run_a_raw_first",
            ["raw", "packed_bytes32"],
            raw,
            packed,
            record.date,
        ),
        run_deployment(
            w3,
            account,
            artifact["abi"],
            artifact["bytecode"],
            "run_b_packed_first",
            ["packed_bytes32", "raw"],
            raw,
            packed,
            record.date,
        ),
    ]
    balance_after = w3.eth.get_balance(account.address)

    read_rows = [row for run in runs for row in run["read_rows"]]
    with (HERE / "read_call_timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(read_rows[0]))
        writer.writeheader()
        writer.writerows(read_rows)

    writes = [write for run in runs for write in run["writes"]]
    by_condition = {
        condition: [write for write in writes if write["condition"] == condition]
        for condition in ("raw", "packed_bytes32")
    }
    gas = {condition: [int(item["receipt"]["gasUsed"]) for item in items] for condition, items in by_condition.items()}
    raw_gas = statistics.mean(gas["raw"])
    packed_gas = statistics.mean(gas["packed_bytes32"])
    read_summary = {}
    for condition in ("raw", "packed_bytes32"):
        values = [row["rpc_call_seconds"] for row in read_rows if row["condition"] == condition]
        estimates = sorted({row["estimated_call_gas"] for row in read_rows if row["condition"] == condition})
        read_summary[condition] = {"estimated_call_gas_values": estimates, **timing_summary(values)}

    summary = {
        "verification_version": "raw-vs-bytes32-opposite-order-1.0",
        "network": "Sepolia",
        "chain_id": CHAIN_ID,
        "started_utc": started,
        "completed_utc": utc_now(),
        "account_address": account.address,
        "credentials_redacted": True,
        "source_record": {
            "date": record.date,
            "text": source_text,
            "raw_bytes": len(raw),
            "raw_sha256": sha256(raw),
            "packed_bytes": len(packed),
            "packed_hex": "0x" + packed.hex(),
            "packed_sha256": sha256(packed),
            "packed_round_trip_exact": format_text_record(unpack_record(packed)) == source_text,
            "size_reduction_bytes": len(raw) - len(packed),
            "size_reduction_percent": (len(raw) - len(packed)) / len(raw) * 100,
        },
        "design": {
            "two_fresh_contract_deployments": True,
            "opposite_write_orders": True,
            "same_date_key_in_separate_raw_and_packed_mappings": True,
            "no_overwrites": True,
            "read_repetitions_per_condition_per_deployment": READ_REPETITIONS_PER_CONDITION_PER_DEPLOYMENT,
        },
        "deployments": [
            {
                "run": run["run"],
                "order": run["order"],
                "contract_address": run["contract_address"],
                "transaction_hash": run["deployment"]["transaction"]["hash"],
                "status": run["deployment"]["receipt"]["status"],
                "gas_used": run["deployment"]["receipt"]["gasUsed"],
                "runtime_code_bytes": run["deployment"]["runtime_code_bytes"],
                "owner_matches_sender": run["deployment"]["owner_matches_sender"],
            }
            for run in runs
        ],
        "writes": [
            {
                "run": item["run"],
                "sequence": item["sequence"],
                "condition": item["condition"],
                "contract_address": next(run["contract_address"] for run in runs if run["run"] == item["run"]),
                "transaction_hash": item["transaction"]["hash"],
                "block_number": item["receipt"]["blockNumber"],
                "payload_bytes": item["payload_bytes"],
                "status": item["receipt"]["status"],
                "gas_used": item["receipt"]["gasUsed"],
                "effective_gas_price_gwei": item["effective_gas_price_gwei"],
                "transaction_fee_eth": item["transaction_fee_eth"],
                "confirmation_seconds": item["confirmation_seconds"],
                "calldata_bytes": item["calldata"]["bytes"],
                "event_count": item["event_count"],
                "event_matches_input": item["event_matches_input"],
                "get_by_date_matches_input": item["get_by_date_matches_input"],
                "packed_decode_matches_source_text": item["packed_decode_matches_source_text"],
                "verification_passed": item["verification_passed"],
            }
            for item in writes
        ],
        "write_comparison": {
            "raw_gas_values": gas["raw"],
            "packed_gas_values": gas["packed_bytes32"],
            "raw_mean_gas": raw_gas,
            "packed_mean_gas": packed_gas,
            "gas_saving": raw_gas - packed_gas,
            "gas_saving_percent_of_raw": (raw_gas - packed_gas) / raw_gas * 100,
        },
        "read_comparison": read_summary,
        "balance_before_wei": balance_before,
        "balance_after_wei": balance_after,
        "total_balance_change_wei_including_deployments": balance_before - balance_after,
        "software": {
            "python": sys.version.split()[0],
            "web3.py": __import__("web3").__version__,
            "platform": platform.platform(),
            "solidity": artifact["compiler"],
        },
        "all_acceptance_checks_passed": all(item["verification_passed"] for item in writes),
    }
    (HERE / "verification_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
