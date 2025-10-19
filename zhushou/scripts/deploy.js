const fs = require('fs');
const path = require('path');
const { ethers } = require('hardhat');

/**
 * Deploy ClimateDataRegistry contract and save address to file.
 * Designed for local Hardhat network (PoA).
 */
async function main() {
    console.log('='.repeat(60));
    console.log('Deploying ClimateDataRegistry Contract');
    console.log('='.repeat(60));

    // Get deployer account
    const [deployer] = await ethers.getSigners();
    console.log(`Deployer address: ${deployer.address}`);
    
    const balance = await ethers.provider.getBalance(deployer.address);
    console.log(`Deployer balance: ${ethers.formatEther(balance)} ETH`);

    // Deploy contract
    console.log('\nDeploying contract...');
    const ClimateDataRegistry = await ethers.getContractFactory('ClimateDataRegistry');
    const registry = await ClimateDataRegistry.deploy();
    
    await registry.waitForDeployment();
    const contractAddress = await registry.getAddress();
    
    console.log(`✓ Contract deployed to: ${contractAddress}`);

    // Save contract address to file
    const contractsDir = path.join(__dirname, '..', 'contracts');
    const addressFile = path.join(contractsDir, 'contract-address.json');

    const deploymentData = {
        address: contractAddress,
        deployer: deployer.address,
        network: 'localhost',
        timestamp: new Date().toISOString(),
        blockNumber: await ethers.provider.getBlockNumber()
    };

    fs.writeFileSync(
        addressFile,
        JSON.stringify(deploymentData, null, 2)
    );

    console.log(`✓ Contract address saved to: ${addressFile}`);
    console.log('\nDeployment Summary:');
    console.log(JSON.stringify(deploymentData, null, 2));
    console.log('='.repeat(60));
}

main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error('Deployment failed:');
        console.error(error);
        process.exit(1);
    });

