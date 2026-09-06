"""Compile the comparison contract with Solidity 0.8.20."""

from __future__ import annotations

import json
import os
from pathlib import Path

import solcx


SOLC_VERSION = "0.8.20"
HERE = Path(__file__).resolve().parent


def main() -> None:
    build_info_path = Path(
        os.environ.get(
            "REMIX_BUILD_INFO",
            r"D:\Remix\workplace\contract\artifacts\build-info\afe4ff7e22c91a30b08ca88c3b7de2d0.json",
        )
    )
    prior = json.loads(build_info_path.read_text(encoding="utf-8"))
    prior_sources = prior["input"]["sources"]
    sources = {
        "contract/eth_store_packed.sol": {
            "content": (HERE / "eth_store_packed.sol").read_text(encoding="utf-8")
        },
        "@openzeppelin/contracts/access/Ownable.sol": prior_sources[
            "@openzeppelin/contracts/access/Ownable.sol"
        ],
        "@openzeppelin/contracts/utils/Context.sol": prior_sources[
            "@openzeppelin/contracts/utils/Context.sol"
        ],
    }
    if SOLC_VERSION not in {str(version) for version in solcx.get_installed_solc_versions()}:
        solcx.install_solc(SOLC_VERSION)
    compiled = solcx.compile_standard(
        {
            "language": "Solidity",
            "sources": sources,
            "settings": {
                "optimizer": {"enabled": False, "runs": 200},
                "outputSelection": {
                    "*": {"*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"]}
                },
            },
        },
        solc_version=SOLC_VERSION,
    )
    contract = compiled["contracts"]["contract/eth_store_packed.sol"][
        "DailyDataStorageComparison"
    ]
    artifact = {
        "contractName": "DailyDataStorageComparison",
        "compiler": SOLC_VERSION,
        "abi": contract["abi"],
        "bytecode": "0x" + contract["evm"]["bytecode"]["object"],
        "deployedBytecode": "0x" + contract["evm"]["deployedBytecode"]["object"],
    }
    (HERE / "contract_artifact.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "contract_abi.json").write_text(
        json.dumps(contract["abi"], indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "compiler": SOLC_VERSION,
                "contract": artifact["contractName"],
                "creation_bytecode_bytes": (len(artifact["bytecode"]) - 2) // 2,
                "runtime_bytecode_bytes": (len(artifact["deployedBytecode"]) - 2) // 2,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
