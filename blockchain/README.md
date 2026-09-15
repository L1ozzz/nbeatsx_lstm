# Sepolia blockchain storage comparison

This directory contains the code and supporting evidence used to compare two representations of one daily environmental record on Ethereum Sepolia:

- a variable-length `bytes` record containing the original text;
- a fixed-width, losslessly encoded `bytes32` record.

## Contract deployments

The following addresses are **two independent deployments of the same `DailyDataStorageComparison` contract**:

- Deployment A: [`0x992114A8C3E975B003209ad5045d3c956E154901`](https://sepolia.etherscan.io/address/0x992114A8C3E975B003209ad5045d3c956E154901)
- Deployment B: [`0x7375A26C9c65B1574eCDbb8C6aa5121f70D4400b`](https://sepolia.etherscan.io/address/0x7375A26C9c65B1574eCDbb8C6aa5121f70D4400b)

They are not two different contract designs. The Solidity source in `eth_store_packed.sol` defines separate `bytes` and `bytes32` mappings, write functions, read functions, and events inside one comparison contract. The two fresh deployments were used as independent experimental runs.

## Main files

- `eth_store_packed.sol`: Solidity source for both storage representations.
- `eth-store.py`: write client for raw and packed records.
- `eth-read.py`: read and verification client.
- `packing.py`: lossless 32-byte encoder and decoder.
- `compile_contract.py`: compiles the contract and generates `contract_abi.json` and `contract_artifact.json`.
- `run_bytes32_comparison.py`: end-to-end comparison runner.
- `read_call_timings.csv`: recorded read-call timing measurements.

## Reproduction

Use Python 3.10 or later and install the required packages:

```text
pip install web3 requests py-solc-x
```

The Solidity source imports OpenZeppelin `Ownable.sol`. Before compiling, set `REMIX_BUILD_INFO` to a Remix build-info JSON file containing the OpenZeppelin sources, then run:

```text
python compile_contract.py
```

This generates `contract_abi.json` beside the clients. Generated ABI, artifact, receipt, and client-output JSON files are intentionally not committed.

For read-only verification, set `SEPOLIA_RPC_URL` and `CONTRACT_ADDRESS`, then run, for example:

```text
python eth-read.py --date 20240523 --mode raw
python eth-read.py --date 20240523 --mode packed
```

For a write, also set `ETH_PRIVATE_KEY` in the process environment and use an address whose contract owner is the corresponding account:

```text
python eth-store.py --mode raw --payload-file source_record.txt
python eth-store.py --mode packed --payload-file source_record.txt
```

Use a fresh date key or a fresh deployment for each write, because the contract rejects overwriting an existing record.

## Credential safety

No private key, mnemonic, seed phrase, or authenticated RPC credential is included in this repository. Supply credentials only through local environment variables and never commit generated output containing sensitive configuration.
