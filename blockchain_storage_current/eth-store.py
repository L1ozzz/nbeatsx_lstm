"""Store the raw or bytes32-packed weather record under its date on Sepolia.

Required environment variables: SEPOLIA_RPC_URL, ETH_PRIVATE_KEY,
CONTRACT_ADDRESS.  The ABI is read from contract_abi.json beside this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from web3 import Web3

from packing import assert_round_trip, parse_text_record


CHAIN_ID = 11155111


def make_web3(url: str) -> Web3:
    session = requests.Session()
    session.trust_env = False
    return Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 60}, session=session))


def fee_fields(w3: Web3) -> dict[str, int]:
    latest = w3.eth.get_block("latest")
    base_fee = int(latest.get("baseFeePerGas") or w3.eth.gas_price)
    priority = w3.to_wei(2, "gwei")
    return {"maxPriorityFeePerGas": priority, "maxFeePerGas": base_fee * 2 + priority}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("raw", "packed"))
    parser.add_argument("--payload-file", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    text = args.payload_file.read_text(encoding="utf-8").strip()
    record = parse_text_record(text)
    raw = text.encode("utf-8")
    packed = assert_round_trip(text)
    payload = raw if args.mode == "raw" else packed

    abi = json.loads(Path(__file__).with_name("contract_abi.json").read_text(encoding="utf-8"))
    w3 = make_web3(os.environ["SEPOLIA_RPC_URL"])
    if not w3.is_connected() or w3.eth.chain_id != CHAIN_ID:
        raise RuntimeError("The configured endpoint is not connected to Sepolia")
    account = w3.eth.account.from_key(os.environ["ETH_PRIVATE_KEY"])
    address = Web3.to_checksum_address(os.environ["CONTRACT_ADDRESS"])
    if not bytes(w3.eth.get_code(address)):
        raise RuntimeError("CONTRACT_ADDRESS has no deployed code")
    contract = w3.eth.contract(address=address, abi=abi)
    if Web3.to_checksum_address(contract.functions.owner().call()) != Web3.to_checksum_address(account.address):
        raise RuntimeError("The signing account is not the contract owner")

    if args.mode == "raw":
        exists = contract.functions.rawDataExists(record.date).call()
        call = contract.functions.storeRawData(record.date, raw)
        event_type = contract.events.RawDataStored()
        event_field = "data"
    else:
        exists = contract.functions.packedDataExists(record.date).call()
        call = contract.functions.storePackedData(record.date, packed)
        event_type = contract.events.PackedDataStored()
        event_field = "data"
    if exists:
        raise RuntimeError(f"{args.mode} data already exists for {record.date}")

    tx = call.build_transaction(
        {
            "from": account.address,
            "nonce": w3.eth.get_transaction_count(account.address, "pending"),
            "chainId": CHAIN_ID,
            **fee_fields(w3),
        }
    )
    estimate = int(w3.eth.estimate_gas(tx))
    tx["gas"] = max(estimate + 20_000, int(estimate * 1.20))
    started = time.perf_counter()
    signed = account.sign_transaction(tx)
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=600)
    elapsed = time.perf_counter() - started
    events = event_type.process_receipt(receipt)
    event_matches = (
        len(events) == 1
        and int(events[0]["args"]["date"]) == record.date
        and bytes(events[0]["args"][event_field]) == payload
    )
    returned = bytes(
        contract.functions.getRawData(record.date).call()
        if args.mode == "raw"
        else contract.functions.getPackedData(record.date).call()
    )
    passed = receipt.status == 1 and event_matches and returned == payload
    result = {
        "network": "Sepolia",
        "contract_address": address,
        "mode": args.mode,
        "date": record.date,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "transaction_hash": w3.to_hex(tx_hash),
        "block_number": receipt.blockNumber,
        "status": receipt.status,
        "gas_used": receipt.gasUsed,
        "effective_gas_price_wei": receipt.effectiveGasPrice,
        "transaction_fee_wei": receipt.gasUsed * receipt.effectiveGasPrice,
        "elapsed_seconds": elapsed,
        "event_count": len(events),
        "event_matches_input": event_matches,
        "readback_matches_input": returned == payload,
        "verification_passed": passed,
        "credentials_redacted": True,
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
