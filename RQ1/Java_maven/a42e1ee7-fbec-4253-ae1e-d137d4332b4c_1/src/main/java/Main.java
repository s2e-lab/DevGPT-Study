public class Main {
    public static void main(string [] args) {
        trigger AccountTrigger on Account (before insert) {
            AccountDuplicateChecker.checkForDuplicates(Trigger.new);
        }
   }
}
