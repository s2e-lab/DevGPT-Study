public class Main {
    public static void main(String[] args) throws Exception {
        DynamicExecutor executor = ExecutorClassGenerator.createExecutorFor(SampleSubscriber.class);
        
        Event<String> event = new Event<>("Hello, ASM!");
        executor.execute(event, new Object[]{"ASM Data", 5});
    }
}
