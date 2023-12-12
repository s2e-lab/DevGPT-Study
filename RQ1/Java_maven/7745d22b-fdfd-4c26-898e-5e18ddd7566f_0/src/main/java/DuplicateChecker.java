// クラス定義
public class DuplicateChecker {

    // 重複をチェックするメソッド
    public static Boolean hasDuplicate(Account account) {
        // チェック対象のフィールドを取得
        String accountName = account.Name;
        String phoneNumber = account.Phone;
        String emailAddress = account.Email;

        // 名前、電話番号、メールアドレスがいずれかのフィールドと一致する取引先を検索するクエリ
        List<Account> duplicates = [SELECT Id
                                    FROM Account
                                    WHERE (Name = :accountName OR Phone = :phoneNumber OR Email = :emailAddress)
                                    AND Id != :account.Id
                                    LIMIT 1];

        // 重複があればtrueを返す
        return !duplicates.isEmpty();
    }
}
