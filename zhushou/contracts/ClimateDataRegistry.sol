// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ClimateDataRegistry
 * @dev Stores provenance metadata for climate data files stored on IPFS.
 * Design principle: On-chain metadata only (hash, source, license, CID).
 * Large files stored off-chain on IPFS for cost-efficiency and scalability.
 */
contract ClimateDataRegistry {
    struct Record {
        bytes32 dataHash;      // SHA-256 hash of the file content
        string sourceURL;      // Original data source URL
        string license;        // License identifier (e.g., CC-BY-4.0)
        uint256 timestamp;     // Block timestamp when recorded
        string ipfsCID;        // IPFS Content Identifier
        address uploader;      // Address that registered this record
    }

    // Storage
    Record[] private records;
    mapping(bytes32 => bool) private hashExists;
    
    // Events
    event RecordAdded(
        uint256 indexed recordId,
        bytes32 indexed dataHash,
        string ipfsCID,
        address indexed uploader,
        uint256 timestamp
    );

    /**
     * @dev Add a new data record to the registry.
     * @param dataHash SHA-256 hash of the file content
     * @param sourceURL Original source URL or description
     * @param license License identifier
     * @param ipfsCID IPFS Content Identifier where file is stored
     * @return id The record ID (index in the records array)
     */
    function addRecord(
        bytes32 dataHash,
        string calldata sourceURL,
        string calldata license,
        string calldata ipfsCID
    ) external returns (uint256 id) {
        require(dataHash != bytes32(0), "Invalid hash");
        require(bytes(ipfsCID).length > 0, "IPFS CID required");
        require(!hashExists[dataHash], "Hash already registered");

        Record memory newRecord = Record({
            dataHash: dataHash,
            sourceURL: sourceURL,
            license: license,
            timestamp: block.timestamp,
            ipfsCID: ipfsCID,
            uploader: msg.sender
        });

        records.push(newRecord);
        hashExists[dataHash] = true;
        id = records.length - 1;

        emit RecordAdded(id, dataHash, ipfsCID, msg.sender, block.timestamp);
        return id;
    }

    /**
     * @dev Retrieve a record by its ID.
     * @param id The record ID
     * @return The record struct
     */
    function getRecord(uint256 id) external view returns (Record memory) {
        require(id < records.length, "Record does not exist");
        return records[id];
    }

    /**
     * @dev Get total number of records in the registry.
     * @return The count of records
     */
    function getRecordCount() external view returns (uint256) {
        return records.length;
    }

    /**
     * @dev Check if a hash has already been registered.
     * @param dataHash The hash to check
     * @return True if hash exists
     */
    function isHashRegistered(bytes32 dataHash) external view returns (bool) {
        return hashExists[dataHash];
    }

    /**
     * @dev Get all records (use with caution, gas-intensive for many records).
     * @return Array of all records
     */
    function getAllRecords() external view returns (Record[] memory) {
        return records;
    }
}

