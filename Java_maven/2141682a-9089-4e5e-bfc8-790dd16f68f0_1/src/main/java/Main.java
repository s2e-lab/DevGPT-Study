class BankAccount {
    private double balance;
    
    private void deductFees() {
        // Some code to deduct fees from the balance
    }
    
    public void deposit(double amount) {
        balance += amount;
    }
    
    public double getBalance() {
        return balance;
    }
}

public class Main {
    public static void main(String[] args) {
        BankAccount account = new BankAccount();
        account.deposit(1000); // Accessing public method to deposit money
        // account.balance = 500; // Error: balance is private and not accessible
        // account.deductFees(); // Error: deductFees() is private and not accessible
    }
}
