"""Independently verify both deployments and all four writes via public RPC."""

from __future__ import annotations

import json
from pathlib import Path

import requests
from web3 import Web3

from packing import assert_round_trip, format_text_record, unpack_record


HERE = Path(__file__).resolve().parent
PUBLIC_RPC = "https://sepolia.drpc.org"


def main() -> None:
    summary = json.loads((HERE / "verification_summary.json").read_text(encoding="utf-8"))
    abi = json.loads((HERE / "contract_abi.json").read_text(encoding="utf-8"))
    source_text = (HERE / "source_record.txt").read_text(encoding="utf-8").strip()
    raw = source_text.encode("utf-8")
    packed = assert_round_trip(source_text)
    date = int(source_text.split(",", 1)[0])

    session = requests.Session()
    session.trust_env = False
    w3 = Web3(Web3.HTTPProvider(PUBLIC_RPC, request_kwargs={"timeout": 60}, session=session))
    if not w3.is_connected() or w3.eth.chain_id != 11155111:
        raise RuntimeError("Public endpoint is not connected to Sepolia")

    deployment_checks = []
    for deployment in summary["deployments"]:
        address = Web3.to_checksum_address(deployment["contract_address"])
        receipt = w3.eth.get_transaction_receipt(deployment["transaction_hash"])
        code = bytes(w3.eth.get_code(address))
        contract = w3.eth.contract(address=address, abi=abi)
        raw_returned = bytes(contract.functions.getRawData(date).call())
        packed_returned = bytes(contract.functions.getPackedData(date).call())
        deployment_checks.append(
            {
                "run": deployment["run"],
                "contract_address": address,
                "deployment_status": receipt.status,
                "runtime_code_nonempty": bool(code),
                "runtime_code_bytes": len(code),
                "raw_exists": contract.functions.rawDataExists(date).call(),
                "packed_exists": contract.functions.packedDataExists(date).call(),
                "raw_get_by_date_matches": raw_returned == raw,
                "packed_get_by_date_matches": packed_returned == packed,
                "packed_decode_matches_source": format_text_record(unpack_record(packed_returned)) == source_text,
            }
        )

    write_checks = []
    for write in summary["writes"]:
        receipt = w3.eth.get_transaction_receipt(write["transaction_hash"])
        tx = w3.eth.get_transaction(write["transaction_hash"])
        write_checks.append(
            {
                "run": write["run"],
                "condition": write["condition"],
                "transaction_hash": write["transaction_hash"],
                "status": receipt.status,
                "gas_used": receipt.gasUsed,
                "gas_matches_private_rpc_record": receipt.gasUsed == write["gas_used"],
                "calldata_bytes": len(bytes(tx.input)),
                "calldata_bytes_matches_private_rpc_record": len(bytes(tx.input)) == write["calldata_bytes"],
                "log_count": len(receipt.logs),
            }
        )

    passed = all(
        item["deployment_status"] == 1
        and item["runtime_code_nonempty"]
        and item["raw_exists"]
        and item["packed_exists"]
        and item["raw_get_by_date_matches"]
        and item["packed_get_by_date_matches"]
        and item["packed_decode_matches_source"]
        for item in deployment_checks
    ) and all(
        item["status"] == 1
        and item["gas_matches_private_rpc_record"]
        and item["calldata_bytes_matches_private_rpc_record"]
        and item["log_count"] == 1
        for item in write_checks
    )
    result = {
        "public_rpc": PUBLIC_RPC,
        "network": "Sepolia",
        "chain_id": w3.eth.chain_id,
        "deployment_checks": deployment_checks,
        "write_checks": write_checks,
        "independent_verification_passed": passed,
    }
    (HERE / "independent_public_rpc_check.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
