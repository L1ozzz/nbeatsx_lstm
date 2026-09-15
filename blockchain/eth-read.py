"""Read and verify a raw or bytes32-packed weather record by YYYYMMDD date."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests
from web3 import Web3

from packing import format_text_record, unpack_record


CHAIN_ID = 11155111


def make_web3(url: str) -> Web3:
    session = requests.Session()
    session.trust_env = False
    return Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 60}, session=session))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, type=int)
    parser.add_argument("--mode", required=True, choices=("raw", "packed"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    w3 = make_web3(os.environ["SEPOLIA_RPC_URL"])
    if not w3.is_connected() or w3.eth.chain_id != CHAIN_ID:
        raise RuntimeError("The configured endpoint is not connected to Sepolia")
    address = Web3.to_checksum_address(os.environ["CONTRACT_ADDRESS"])
    if not bytes(w3.eth.get_code(address)):
        raise RuntimeError("CONTRACT_ADDRESS has no deployed code")
    abi = json.loads(Path(__file__).with_name("contract_abi.json").read_text(encoding="utf-8"))
    contract = w3.eth.contract(address=address, abi=abi)

    if args.mode == "raw":
        exists = contract.functions.rawDataExists(args.date).call()
        call = contract.functions.getRawData(args.date)
    else:
        exists = contract.functions.packedDataExists(args.date).call()
        call = contract.functions.getPackedData(args.date)
    if not exists:
        raise RuntimeError(f"No {args.mode} record exists for {args.date}")

    started = time.perf_counter()
    returned = bytes(call.call())
    elapsed = time.perf_counter() - started
    estimated_gas = int(call.estimate_gas())
    if args.mode == "raw":
        decoded_text = returned.decode("utf-8")
        embedded_date = int(decoded_text.split(",", 1)[0])
    else:
        decoded = unpack_record(returned)
        decoded_text = format_text_record(decoded)
        embedded_date = decoded.date

    result = {
        "network": "Sepolia",
        "contract_address": address,
        "date_key": args.date,
        "mode": args.mode,
        "data_exists": exists,
        "returned_bytes": len(returned),
        "returned_sha256": hashlib.sha256(returned).hexdigest(),
        "estimated_call_gas": estimated_gas,
        "rpc_call_seconds": elapsed,
        "decoded_text": decoded_text,
        "embedded_date": embedded_date,
        "embedded_date_matches_key": embedded_date == args.date,
        "read_by_date_passed": embedded_date == args.date,
    }
    output = json.dumps(result, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    if not result["read_by_date_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
