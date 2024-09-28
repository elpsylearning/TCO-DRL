pragma solidity ^0.5.12;

contract Selection {
    uint action;
    
    function ReAction(uint ac) public returns(uint){
        action=ac;
        uint ac_d=action;
        return ac_d;
    }

    function SeAction() public view returns(uint){
       uint RecommandAction = action;
       return RecommandAction;
    }
}
