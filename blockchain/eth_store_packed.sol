// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/Ownable.sol";

/// @title Side-by-side storage of raw and fixed-width environmental records.
/// @notice The two mappings are separate so the same YYYYMMDD key can be used
///         once in each condition without either write becoming an overwrite.
contract DailyDataStorageComparison is Ownable(msg.sender) {
    mapping(uint256 => bytes) private rawRecords;
    mapping(uint256 => bytes32) private packedRecords;

    event RawDataStored(uint256 indexed date, bytes data);
    event PackedDataStored(uint256 indexed date, bytes32 data);

    function storeRawData(uint256 date, bytes calldata data) external onlyOwner {
        require(date != 0, "Date must not be zero.");
        require(data.length != 0, "Data must not be empty.");
        require(!rawDataExists(date), "Raw data already exists.");
        rawRecords[date] = data;
        emit RawDataStored(date, data);
    }

    function storePackedData(uint256 date, bytes32 data) external onlyOwner {
        require(date != 0, "Date must not be zero.");
        require(data != bytes32(0), "Data must not be empty.");
        require(!packedDataExists(date), "Packed data already exists.");
        packedRecords[date] = data;
        emit PackedDataStored(date, data);
    }

    function getRawData(uint256 date) external view returns (bytes memory) {
        require(rawDataExists(date), "Raw data does not exist.");
        return rawRecords[date];
    }

    function getPackedData(uint256 date) external view returns (bytes32) {
        require(packedDataExists(date), "Packed data does not exist.");
        return packedRecords[date];
    }

    function rawDataExists(uint256 date) public view returns (bool) {
        return rawRecords[date].length != 0;
    }

    function packedDataExists(uint256 date) public view returns (bool) {
        return packedRecords[date] != bytes32(0);
    }
}
