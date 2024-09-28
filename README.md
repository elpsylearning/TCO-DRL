# TCO-DRL
TCO-DRL is implemented in two versions. The ‘TCO-DRL_with baseline ’ compares the performance of TCO-DRL with several baseline methods; you can directly run `main.py` after configuring the environment. The ‘TCO-DRL_on blockchain’ is deployed separately on Ethereum for experimental testing, with specific configurations and steps outlined below.
## Configuration
 - Ubuntu 18.04
-  Geth 1.10.25
-  Go 1.18.5
-  Truffle 5.5.32
-  Node 16.17.1
-  Python 3.6.9
-  Tensorflow 2.0.0
-  Ganache-cli
-  Web3
## Steps
### **1. Accounts generation**
```
ganache-cli -a 16
```
**Modify:**  Number of accounts. Generate a valid account for each oracle and processor.

### **2. Smart Contracts Compilation**
```
truffle compile
```
**Note:**  Open a new terminal and run this command.

### **3. Smart Contracts Migration**
```
truffle migrate
```
**Note:**  Record the contract address of the smart contracts.

### **4. Trust-Aware and Cost-Optimized Blockchain Oracle Selection**
```
cd pythonProject
```

```
sudo python main.py
```

**Modify:**  

> w3(Web3.HTTPProvider)
> SELECTION_CONTRACT_ADDR
> self.web3(Web3.WebsocketProvider)
> the path of compiled smart contracts

