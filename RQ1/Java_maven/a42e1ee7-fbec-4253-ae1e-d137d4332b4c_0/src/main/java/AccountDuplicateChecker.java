public class AccountDuplicateChecker {
    public static void checkForDuplicates(List<Account> newAccounts) {
        Set<String> nameSet = new Set<String>();
        Set<String> phoneSet = new Set<String>();
        Set<String> emailSet = new Set<String>();

        for(Account a : newAccounts) {
            nameSet.add(a.Name);
            phoneSet.add(a.Phone);
            emailSet.add(a.Email__c);
        }

        List<Account> existingAccounts = [SELECT Name, Phone, Email__c FROM Account WHERE Name IN :nameSet OR Phone IN :phoneSet OR Email__c IN :emailSet];

        for(Account a : newAccounts) {
            for(Account existing : existingAccounts) {
                if(a.Name == existing.Name || a.Phone == existing.Phone || a.Email__c == existing.Email__c) {
                    // ここではエラーメッセージを生成しますが、代わりに別の操作を行うことも可能です
                    a.addError('取引先が重複しています。');
                }
            }
        }
    }
}
